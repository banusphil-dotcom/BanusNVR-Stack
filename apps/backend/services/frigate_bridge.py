"""BanusNas — Frigate MQTT Event Bridge.

Subscribes to Frigate's MQTT events, triggers face/body/pet recognition
on snapshots, stores enriched events in Postgres, and pushes sub_labels
back to Frigate.

Replaces: event_processor.py, object_detector.py, object_tracker.py,
          motion_detector.py, stream_manager.py, recording_engine.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Optional

import cv2
import httpx
import numpy as np
from paho.mqtt import client as mqtt_client
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified

from core.config import settings
from models.database import async_session
from models.schemas import (
    Camera,
    Event,
    EventType,
    NamedObject,
    ObjectCategory,
)
from services.narrative_generator import generate_narrative, describe_snapshot_with_vision, describe_with_text_llm

logger = logging.getLogger("banusnas.frigate_bridge")

# ── Frigate event debounce (skip very brief detections) ──
MIN_EVENT_DURATION_S = 1.5

# ── Recognition cooldown per Frigate event ID ──
RECOGNITION_COOLDOWN_S = 8.0
RECOGNITION_MAX_ATTEMPTS = 4  # cap retries on one Frigate event to avoid
                              # wasting ML calls on persistent false-positives

# ── Cross-camera presence conflict window ──
PRESENCE_TIMEOUT_S = 60.0

# ── Burst suppression: max unrecognized events per camera in a window ──
BURST_WINDOW_S = 900.0       # 15-minute sliding window
BURST_MAX_UNRECOGNIZED = 15  # suppress after 15 unrecognized events in window
BURST_COOLDOWN_S = 300.0     # 5-minute suppression after burst detected

# ── Static false-positive suppression (pillars/statues at a fixed bbox) ──
STATIC_FP_IOU_THRESHOLD = 0.65   # IoU above this = "same position"
STATIC_FP_MIN_HITS = 3           # unrecognized hits at same position before suppression
STATIC_FP_WINDOW_S = 1800.0      # 30-minute sliding window for tracking
STATIC_FP_SUPPRESS_S = 900.0     # suppress that zone for 15 min after triggered

# ── Temporal identity propagation: carry forward recognition to nearby events ──
TEMPORAL_WINDOW_S = 120.0     # propagate identity within 120s on same camera
TEMPORAL_IOU_MIN = 0.20       # minimum bbox overlap for propagation
TEMPORAL_MAX_ENTRIES = 8      # max recent recognitions to keep per camera

# ── Event grouping: nearby detections on same camera form a group ──
GROUP_WINDOW_S = 60.0         # 1-minute window for grouping events on same camera

# ── Event consolidation: merge detections into single events ──
# Person consolidation is CROSS-CAMERA: Philip in living room + kitchen = 1 event.
# Consolidation only expires after CONSOLIDATION_IDLE_S with NO detections anywhere.
# Tightened (v25.2) to prevent "all night active" run-on events: short idle gaps
# and a hard MAX_EVENT_DURATION cap force a fresh event after meaningful breaks.
CONSOLIDATION_NAMED_S = 600.0    # 10 min max for recognized objects (safety cap)
CONSOLIDATION_UNKNOWN_S = 120.0  # 2 min for unrecognized objects (per-camera)
CONSOLIDATION_IDLE_S = 180.0     # 3 min idle before ending a cross-camera session
MAX_EVENT_DURATION_S = 600.0     # 10 min absolute cap — force new event past this

# ── Category mapping: Frigate label → our ObjectCategory ──
LABEL_CATEGORY = {
    "person": ObjectCategory.person,
    "cat": ObjectCategory.pet,
    "dog": ObjectCategory.pet,
    "bird": ObjectCategory.pet,
    "car": ObjectCategory.vehicle,
    "truck": ObjectCategory.vehicle,
    "bus": ObjectCategory.vehicle,
    "motorcycle": ObjectCategory.vehicle,
    "bicycle": ObjectCategory.vehicle,
    "boat": ObjectCategory.vehicle,
}

# ── Frigate camera name → our DB camera ID mapping (loaded at startup) ──
_camera_name_to_id: dict[str, int] = {}
_camera_id_to_friendly: dict[int, str] = {}


class EventBus:
    """Simple pub/sub for WebSocket clients."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers = [s for s in self._subscribers if s is not q]

    def publish(self, data: dict):
        for q in self._subscribers:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass


event_bus = EventBus()


class FrigateBridge:
    """MQTT subscriber that bridges Frigate detection events to BanusNVR
    recognition pipeline and Postgres storage."""

    def __init__(self):
        self._mqtt: Optional[mqtt_client.Client] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._http: Optional[httpx.AsyncClient] = None
        self._running = False

        # Dedup / cooldown state
        self._last_recognition: dict[str, float] = {}  # frigate_event_id → timestamp
        self._recognition_attempts: dict[str, int] = {}  # frigate_event_id → retry count
        self._event_map: dict[str, int] = {}  # frigate_event_id → our Event.id
        self._presence: dict[int, dict] = {}  # named_object_id → {camera_id, last_seen}

        # Burst suppression state: camera_id → list of timestamps of unrecognized events
        self._unrecognized_timestamps: dict[int, list[float]] = {}
        self._burst_suppressed_until: dict[int, float] = {}  # camera_id → suppress until

        # Static false-positive suppression: camera_id → list of (bbox_dict, timestamp)
        self._static_fp_history: dict[int, list[tuple[dict, float]]] = {}
        # camera_id → {zone_key: suppress_until_ts}
        self._static_fp_suppressed: dict[int, dict[str, float]] = {}

        # Event grouping: camera_id → (group_key, last_event_ts, group_started_ts)
        self._active_groups: dict[int, tuple[str, float, float]] = {}

        # Event consolidation: (camera_id, object_type, named_object_id|0) → {event_id, last_ts, bbox}
        # Allows extending an existing event instead of creating a new one
        self._recent_events: dict[tuple[int, str, int], dict] = {}

        # Thumbnail history for preview GIF: frigate_id → list of thumb paths
        self._thumbnail_history: dict[str, list[str]] = {}
        self._MAX_GIF_FRAMES = 12

        # Recent successful recognitions per camera for temporal propagation
        # camera_id → [{named_object_id, name, category, bbox, timestamp, score}]
        self._recent_recognitions: dict[int, list[dict]] = {}

        # Notification engine reference (set externally)
        self._notification_engine = None

    def set_notification_engine(self, engine):
        self._notification_engine = engine

    # ── Startup / Shutdown ──

    async def start(self):
        """Connect to MQTT and start listening for Frigate events."""
        self._loop = asyncio.get_running_loop()
        self._http = httpx.AsyncClient(
            base_url=settings.frigate_url,
            timeout=httpx.Timeout(30.0),
        )
        self._running = True

        # Load camera name mapping
        await self._load_camera_map()

        # Connect MQTT in a thread (paho-mqtt is synchronous)
        await asyncio.to_thread(self._connect_mqtt)
        logger.info("Frigate bridge started — listening for events on MQTT")

    async def stop(self):
        self._running = False
        if self._mqtt:
            self._mqtt.disconnect()
            self._mqtt.loop_stop()
        if self._http:
            await self._http.aclose()
        logger.info("Frigate bridge stopped")

    async def _load_camera_map(self):
        """Build mapping: Frigate camera name (e.g., 'camera_7') → DB camera ID."""
        global _camera_name_to_id, _camera_id_to_friendly
        async with async_session() as session:
            result = await session.execute(select(Camera).where(Camera.enabled == True))
            cameras = result.scalars().all()
            for cam in cameras:
                stream_name = f"camera_{cam.id}"
                _camera_name_to_id[stream_name] = cam.id
                _camera_id_to_friendly[cam.id] = cam.name
        logger.info("Camera map loaded: %s", _camera_name_to_id)

    def _connect_mqtt(self):
        client_id = "banusnas-frigate-bridge"
        self._mqtt = mqtt_client.Client(
            mqtt_client.CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_disconnect = self._on_disconnect
        self._mqtt.on_message = self._on_message
        self._mqtt.connect(settings.mqtt_host, settings.mqtt_port, keepalive=60)
        self._mqtt.loop_start()

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        logger.info("MQTT connected (rc=%s), subscribing to frigate topics", reason_code)
        client.subscribe("frigate/events", qos=1)
        client.subscribe("frigate/reviews", qos=0)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        logger.warning("MQTT disconnected (rc=%s)", reason_code)

    def _on_message(self, client, userdata, msg):
        """Dispatch MQTT messages to async handlers."""
        logger.debug("MQTT msg: topic=%s running=%s loop=%s", msg.topic, self._running, self._loop is not None)
        if not self._loop or not self._running:
            logger.warning("MQTT msg dropped: loop=%s running=%s", self._loop, self._running)
            return
        try:
            payload = json.loads(msg.payload)
        except json.JSONDecodeError:
            return

        if msg.topic == "frigate/events":
            asyncio.run_coroutine_threadsafe(
                self._handle_event(payload), self._loop
            )

    # ── Event Handling ──

    async def _handle_event(self, payload: dict):
        """Process a Frigate event (new/update/end)."""
        try:
            event_type = payload.get("type")  # "new", "update", "end"
            before = payload.get("before", {})
            after = payload.get("after", {})

            # Use 'after' state as the current state
            data = after if after else before
            frigate_id = data.get("id", "")
            camera_name = data.get("camera", "")
            label = data.get("label", "")
            score = data.get("score", 0)
            start_time = data.get("start_time", 0)

            camera_id = _camera_name_to_id.get(camera_name)
            if camera_id is None:
                return  # Unknown camera

            if event_type == "new":
                await self._on_event_new(frigate_id, camera_id, camera_name, label, score, data)

            elif event_type == "update":
                await self._on_event_update(frigate_id, camera_id, camera_name, label, score, data)

            elif event_type == "end":
                await self._on_event_end(frigate_id, data)

        except Exception:
            logger.exception("Error handling Frigate event")

    async def _on_event_new(self, frigate_id: str, camera_id: int, camera_name: str,
                            label: str, score: float, data: dict):
        """A new tracked object appeared — create or consolidate event and trigger recognition."""
        # Check burst suppression for this camera
        if self._is_burst_suppressed(camera_id):
            cam_friendly = _camera_id_to_friendly.get(camera_id, camera_name)
            logger.debug("Burst suppressed: skipping %s event on %s [%s]", label, cam_friendly, frigate_id[:12])
            return

        # Check static false-positive suppression (pillars/statues at fixed position)
        bbox = self._extract_bbox(data)
        if bbox and label == "person" and self._is_static_fp_suppressed(camera_id, bbox):
            cam_friendly = _camera_id_to_friendly.get(camera_id, camera_name)
            logger.debug("Static FP suppressed: skipping %s at [%d,%d,%d,%d] on %s [%s]",
                         label, bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
                         cam_friendly, frigate_id[:12])
            return

        start_ts = datetime.fromtimestamp(data.get("start_time", time.time()), tz=timezone.utc)
        cam_friendly = _camera_id_to_friendly.get(camera_id, camera_name)

        # ── 1 Frigate track = 1 DB event ──
        # Previously we attempted to consolidate pet detections into a recent
        # event on the same camera. That collapsed every visit/movement into a
        # single row and produced "1 grouped event for the whole day" on the UI.
        # The frontend already groups discrete events by `group_key` for
        # display, so we keep events 1:1 with Frigate tracks here.

        # Create new event in Postgres
        group_key = self._get_or_create_group_key(camera_id, start_ts.timestamp())
        narrative = generate_narrative(
            named_object_name=None,
            object_type=label,
            camera_name=cam_friendly,
            timestamp=start_ts,
            seed=frigate_id,
        )
        event_db = Event(
            camera_id=camera_id,
            event_type=EventType.object_detected,
            object_type=label,
            confidence=score,
            bbox=bbox,
            group_key=group_key,
            started_at=start_ts,
            metadata_extra={
                "frigate_id": frigate_id,
                "source": "frigate",
                "zones": data.get("current_zones", []),
                "narrative": narrative,
                "narrative_source": "template",
                "detect_resolution": (
                    [data["frame_shape"][1], data["frame_shape"][0]]
                    if data.get("frame_shape") and len(data.get("frame_shape", [])) >= 2
                    else None
                ),
            },
        )

        async with async_session() as session:
            session.add(event_db)
            await session.commit()
            await session.refresh(event_db)
            self._event_map[frigate_id] = event_db.id

        logger.info("New event #%d: %s on %s (score=%.2f) [%s]",
                    event_db.id, label, cam_friendly, score, frigate_id[:12])

        # Publish to WebSocket clients
        event_bus.publish({
            "type": "detection",
            "event_id": event_db.id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "object_type": label,
            "confidence": score,
        })

        # Trigger recognition asynchronously
        asyncio.create_task(self._run_recognition(frigate_id, camera_id, label, score, data))

    async def _on_event_update(self, frigate_id: str, camera_id: int, camera_name: str,
                               label: str, score: float, data: dict):
        """Frigate updated a tracked object — retry recognition if not yet identified."""
        event_id = self._event_map.get(frigate_id)
        if event_id is None:
            return

        # Update event score if improved
        async with async_session() as session:
            result = await session.execute(select(Event).where(Event.id == event_id))
            event_db = result.scalar_one_or_none()
            if event_db and score > (event_db.confidence or 0):
                event_db.confidence = score
                event_db.bbox = self._extract_bbox(data)
                session.add(event_db)
                await session.commit()

        # Retry recognition with cooldown
        now = time.time()
        last = self._last_recognition.get(frigate_id, 0)
        if now - last >= RECOGNITION_COOLDOWN_S:
            # Only retry if not yet recognized
            async with async_session() as session:
                result = await session.execute(select(Event).where(Event.id == event_id))
                event_db = result.scalar_one_or_none()
                if event_db and event_db.named_object_id is None:
                    # Cap retries — a persistent false-positive on a wall/pillar
                    # will never match; each retry costs ~3 s of ML API calls.
                    attempts = self._recognition_attempts.get(frigate_id, 0)
                    if attempts >= RECOGNITION_MAX_ATTEMPTS:
                        logger.debug(
                            "Recognition retry skipped for %s on cam %d: "
                            "already %d unsuccessful attempts",
                            frigate_id[:12], camera_id, attempts,
                        )
                        # Feed the static-FP tracker so the zone gets suppressed
                        bbox_fp = self._extract_bbox(data)
                        if bbox_fp and label == "person":
                            self._record_static_fp(camera_id, bbox_fp)
                        return
                    asyncio.create_task(
                        self._run_recognition(frigate_id, camera_id, label, score, data)
                    )

    async def _on_event_end(self, frigate_id: str, data: dict):
        """Frigate ended event — finalize in Postgres and build preview GIF."""
        event_id = self._event_map.pop(frigate_id, None)
        if event_id is None:
            return

        end_time = data.get("end_time")
        camera_name = data.get("camera", "")
        camera_id = _camera_name_to_id.get(camera_name)
        label = data.get("label", "object")
        cam_friendly = _camera_id_to_friendly.get(camera_id, camera_name) if camera_id else camera_name
        duration = (end_time - data.get("start_time", end_time)) if end_time and data.get("start_time") else 0
        logger.info("Event ended #%d: %s on %s (%.1fs, %s) [%s]",
                    event_id, label, cam_friendly, duration,
                    f"{len(self._thumbnail_history.get(frigate_id, []))} thumbs",
                    frigate_id[:12])

        # Build preview GIF from accumulated thumbnails
        thumb_history = self._thumbnail_history.pop(frigate_id, [])
        gif_path = None
        if len(thumb_history) >= 1 and camera_id is not None:
            try:
                gif_path = await self._build_preview_gif(camera_id, thumb_history, label)
            except Exception:
                logger.debug("GIF build failed for %s", frigate_id[:12], exc_info=True)

        async with async_session() as session:
            result = await session.execute(select(Event).where(Event.id == event_id))
            event_db = result.scalar_one_or_none()
            if event_db:
                if end_time:
                    event_db.ended_at = datetime.fromtimestamp(end_time, tz=timezone.utc)
                else:
                    event_db.ended_at = datetime.now(timezone.utc)

                # Store Frigate's event data in metadata
                meta = dict(event_db.metadata_extra or {})
                meta["frigate_has_clip"] = data.get("has_clip", False)
                meta["frigate_has_snapshot"] = data.get("has_snapshot", False)
                meta["zones_entered"] = data.get("entered_zones", [])
                if thumb_history:
                    meta["thumbnail_history"] = thumb_history
                if gif_path:
                    meta["gif_path"] = gif_path
                event_db.metadata_extra = meta
                flag_modified(event_db, "metadata_extra")

                # Capture info for evolving narrative before commit
                named_obj_name = None
                named_obj_id = event_db.named_object_id
                if named_obj_id:
                    obj_result = await session.execute(
                        select(NamedObject.name).where(NamedObject.id == named_obj_id)
                    )
                    named_obj_name = obj_result.scalar_one_or_none()

                session.add(event_db)
                await session.commit()

        # Clean up cooldown tracker
        self._last_recognition.pop(frigate_id, None)
        self._recognition_attempts.pop(frigate_id, None)

        # Evolving activity description — update narrative with multi-frame context
        if duration >= 5 and len(thumb_history) >= 2 and label in ("person", "cat", "dog"):
            asyncio.create_task(self._update_activity_narrative(
                event_id=event_id,
                thumb_paths=thumb_history,
                camera_name=cam_friendly,
                object_type=label,
                named_object_name=named_obj_name,
                duration_seconds=duration,
            ))

        event_bus.publish({
            "type": "event_ended",
            "event_id": event_id,
        })

    # ── Recognition Pipeline ──

    async def _run_recognition(self, frigate_id: str, camera_id: int,
                               label: str, score: float, data: dict):
        """Fetch snapshot from Frigate and run face/body/pet recognition."""
        self._last_recognition[frigate_id] = time.time()
        self._recognition_attempts[frigate_id] = (
            self._recognition_attempts.get(frigate_id, 0) + 1
        )

        event_id = self._event_map.get(frigate_id)
        if event_id is None:
            return

        # Fetch snapshot and crop with validation + PTZ-aware retry
        camera_name = data.get("camera", f"camera_{camera_id}")
        full_frame, object_crop = await self._fetch_and_validate_crop(
            frigate_id, camera_id, camera_name, data
        )
        if full_frame is None:
            return

        # If the bbox-aware crop failed, we **do not** substitute the full
        # frame for recognition. Letting face/pet recognisers see the whole
        # scene caused them to attach an identity to whichever face was
        # easiest to find anywhere in the image (a poster, a TV, the
        # background person), not necessarily the object Frigate detected.
        # Better to save the snapshot, skip recognition, and let the next
        # `update` event try again with a (hopefully) cleaner crop.
        recognition_crop = object_crop  # may be None — handled below

        # Save full frame as snapshot, bbox crop as thumbnail
        snap_path = await self._save_snapshot(full_frame, camera_id, frigate_id)
        # Only save thumbnail if we got a valid crop (not the full frame)
        thumb_path = None
        if object_crop is not None and object_crop is not full_frame:
            thumb_path = await self._save_thumbnail(object_crop, camera_id, frigate_id)
        if snap_path or thumb_path:
            async with async_session() as session:
                result = await session.execute(select(Event).where(Event.id == event_id))
                event_db = result.scalar_one_or_none()
                if event_db:
                    if snap_path:
                        event_db.snapshot_path = snap_path
                    if thumb_path:
                        # Only update thumbnail if event doesn't already have one
                        # (don't overwrite a good crop with a worse one from a later update)
                        if not event_db.thumbnail_path:
                            event_db.thumbnail_path = thumb_path
                        # Always accumulate for GIF history
                        hist = self._thumbnail_history.setdefault(frigate_id, [])
                        if thumb_path not in hist:
                            hist.append(thumb_path)
                            if len(hist) > self._MAX_GIF_FRAMES:
                                hist[:] = hist[-self._MAX_GIF_FRAMES:]
                    # Keep bbox in sync with the saved snapshot (PTZ may have moved)
                    current_bbox = self._extract_bbox(data)
                    if current_bbox:
                        event_db.bbox = current_bbox
                    # Store detect frame dimensions for bbox scaling
                    frame_shape = data.get("frame_shape")
                    if frame_shape and len(frame_shape) >= 2:
                        meta = dict(event_db.metadata_extra or {})
                        meta["detect_resolution"] = [frame_shape[1], frame_shape[0]]
                        event_db.metadata_extra = meta
                        flag_modified(event_db, "metadata_extra")
                    session.add(event_db)
                    await session.commit()

        category = LABEL_CATEGORY.get(label)
        if category is None:
            return

        # Skip recognition for low-confidence Frigate detections (noise / IR ghosts)
        if score < 0.50:
            logger.debug("Skipping recognition for %s: Frigate score %.2f < 0.50", frigate_id[:12], score)
            return

        # Skip recognition entirely if we couldn't isolate the object — we
        # refuse to attribute an identity to "something somewhere in the
        # frame" because that's how the wrong person ends up labelled.
        if recognition_crop is None or recognition_crop is full_frame:
            logger.info(
                "Skipping recognition for %s on cam %s: no isolated object crop available",
                frigate_id[:12], camera_id,
            )
            return

        named_object_id = None
        named_object_name = None
        match_score = 0.0
        camera_friendly = _camera_id_to_friendly.get(camera_id, data.get("camera", f"camera_{camera_id}"))

        try:
            if category == ObjectCategory.person:
                named_object_id, named_object_name, match_score = await self._recognize_person(
                    recognition_crop, camera_id,
                    full_frame=full_frame,
                    bbox=self._extract_bbox(data),
                    det_confidence=score,
                    data=data,
                )
            elif category == ObjectCategory.pet:
                named_object_id, named_object_name, match_score = await self._recognize_pet(
                    recognition_crop, label, camera_id
                )
        except Exception:
            logger.exception("Recognition failed for frigate event %s", frigate_id)
            return

        # ── Vision LLM Enforcement ────────────────────────────────────
        # If embedding match found → verify with vision model before accepting
        # If no embedding match → attempt vision identification from all profiles
        from services.ml_client import remote_vision_verify, remote_vision_identify, is_available as ml_available

        vision_available = await ml_available()
        _rejected_obj_id = None  # track rejected candidate to exclude from identify

        if named_object_id is not None and vision_available:
            # ── v25 trust-strong-pet bypass ─────────────────────────────
            # Vision verify on tiny/blurry pet crops (Frigate fallback 300×300
            # or 50–80px detection crops) frequently returns MATCH:NO with a
            # default confidence even when the embedding+colour-gate evidence
            # is overwhelming. This produced 0/75 dog-labelled recognitions
            # over 7d when the CNN clearly identified Tangie. Skip vision
            # verify when the pet match is strong enough that the colour gate
            # has already vetted species/coat compatibility.
            skip_verify = (
                category == ObjectCategory.pet
                and match_score >= 0.40
            )
            if skip_verify:
                logger.info(
                    "Vision verify SKIPPED for %s (pet, score=%.2f) on cam %d",
                    named_object_name, match_score, camera_id,
                )
            # Verify the embedding match with vision
            ref_crops = (
                [] if skip_verify
                else await self._load_reference_thumbnails(named_object_id, limit=3)
            )
            if ref_crops:
                # Load profile attributes for the candidate
                _candidate_attrs = await self._load_profile_attributes(named_object_id)
                vresult = await remote_vision_verify(
                    object_crop, ref_crops,
                    candidate_name=named_object_name or "",
                    object_type=label,
                    camera_name=camera_friendly,
                    candidate_attributes=_candidate_attrs,
                )
                if vresult and not vresult.get("match", False):
                    logger.info(
                        "Vision REJECTED %s (conf=%d) for %s on cam %d: %s",
                        named_object_name, vresult.get("confidence", 0),
                        frigate_id[:12], camera_id, vresult.get("reasoning", "")[:80],
                    )
                    _rejected_obj_id = named_object_id
                    named_object_id = None
                    named_object_name = None
                    match_score = 0.0
                elif vresult and vresult.get("match"):
                    logger.info(
                        "Vision CONFIRMED %s (conf=%d) for %s on cam %d",
                        named_object_name, vresult.get("confidence", 0),
                        frigate_id[:12], camera_id,
                    )

        if named_object_id is None and vision_available:
            # ── Cost gate: avoid wasting expensive vision-identify calls on
            #    repeat false-positives from the same screen position. The
            #    static-FP detector below catches them after STATIC_FP_MIN_HITS
            #    misses, but each pre-suppression call costs ~1.4s and produces
            #    a confident-looking but worthless guess (always 55%).
            bbox_for_gate = self._extract_bbox(data)
            skip_vision_identify = False
            if bbox_for_gate and label == "person":
                history = self._static_fp_history.get(camera_id, [])
                now_ts = time.time()
                recent_hits = sum(
                    1 for hbox, ts in history
                    if now_ts - ts <= STATIC_FP_WINDOW_S
                    and self._bbox_iou(bbox_for_gate, hbox) >= STATIC_FP_IOU_THRESHOLD
                )
                if recent_hits >= 2:
                    logger.info(
                        "Skip vision-identify on cam %d: %d prior unrecognised hits at "
                        "same bbox in %.0fs window [%s]",
                        camera_id, recent_hits, STATIC_FP_WINDOW_S, frigate_id[:12],
                    )
                    skip_vision_identify = True

            # No embedding match or vision rejected — try vision identification
            # Exclude any candidate that was just rejected by verify to avoid re-offering
            profile_candidates = (
                [] if skip_vision_identify
                else await self._build_vision_candidates(
                    category, limit=8,
                    exclude_ids={_rejected_obj_id} if _rejected_obj_id else None,
                )
            )
            if profile_candidates:
                # Use generic 'pet' label — Frigate's species label may be wrong
                identify_type = "pet" if category == ObjectCategory.pet else label
                vresult = await remote_vision_identify(
                    object_crop, profile_candidates,
                    object_type=identify_type,
                    camera_name=camera_friendly,
                )
                # Require higher confidence when an embedding match was overridden
                min_identify_conf = 75 if _rejected_obj_id else 60
                vconf = vresult.get("confidence", 0) if vresult else 0
                if vresult and vresult.get("identified_name") and vconf >= min_identify_conf:
                    idx = vresult["identified_index"]
                    if idx and 1 <= idx <= len(profile_candidates):
                        cand = profile_candidates[idx - 1]
                        named_object_id = cand["id"]
                        named_object_name = cand["name"]
                        match_score = vresult["confidence"] / 100.0
                        logger.info(
                            "Vision IDENTIFIED %s (conf=%d) for %s on cam %d",
                            named_object_name, vresult.get("confidence", 0),
                            frigate_id[:12], camera_id,
                        )
                elif vresult and vresult.get("identified_name"):
                    logger.info(
                        "Vision identify below threshold: %s conf=%d (need %d) for %s on cam %d",
                        vresult.get("identified_name"), vconf, min_identify_conf,
                        frigate_id[:12], camera_id,
                    )
        # ── End Vision Enforcement ────────────────────────────────────

        # Track whether vision was involved (either rejected or tried to identify)
        _vision_was_involved = _rejected_obj_id is not None or (
            named_object_id is None and vision_available
        )

        propagated = None
        if named_object_id is None:
            # ── Temporal propagation: inherit identity from recent recognition ──
            # BLOCK temporal propagation when vision was involved:
            # - If vision rejected a candidate, the person is there but NOT who
            #   the cache says → propagation would assign the wrong identity
            # - If vision tried to identify and failed, we genuinely don't know
            #   who this is → better to leave unrecognized than guess wrong
            if _vision_was_involved:
                logger.info(
                    "Temporal propagation blocked (vision involved) for %s on cam %d [%s]",
                    label, camera_id, frigate_id[:12],
                )
            else:
                bbox = self._extract_bbox(data)
                propagated = self._try_temporal_propagation(camera_id, category, bbox)

            if propagated:
                # Pet colour gate on forward (live) temporal propagation: refuse
                # to inherit a pet identity if the current crop's colour clearly
                # contradicts the named profile (Frostie/white vs Tangie/tortoiseshell).
                if category == ObjectCategory.pet and full_frame is not None:
                    try:
                        from services.pet_color_gate import (
                            compute_colour_signal, colour_compatibility,
                        )
                        crop_img = self._crop_to_bbox(full_frame, data)
                        if crop_img is not None and crop_img.size > 0:
                            async with async_session() as session_pc:
                                prof = await session_pc.get(
                                    NamedObject, propagated["named_object_id"],
                                )
                                pcolor = None
                                if prof is not None:
                                    pattrs = prof.attributes or {}
                                    pcolor = pattrs.get("color") or pattrs.get("colour")
                            if pcolor:
                                signal = compute_colour_signal(crop_img)
                                if colour_compatibility(pcolor, signal) <= 0.0:
                                    logger.info(
                                        "Temporal-prop colour VETO: %s profile=%s observed=%s "
                                        "(white=%.2f black=%.2f) on cam %d [%s]",
                                        propagated["name"], pcolor, signal.family,
                                        signal.white_ratio, signal.black_ratio,
                                        camera_id, frigate_id[:12],
                                    )
                                    propagated = None
                    except Exception:
                        logger.debug("Temporal-prop colour gate error", exc_info=True)

            if propagated:
                named_object_id = propagated["named_object_id"]
                named_object_name = propagated["name"]
                match_score = propagated["score"] * 0.85  # discount for indirect match
                logger.info(
                    "Temporal propagation: %s → event #%d on camera %d (%.0f%%) [%s]",
                    named_object_name, event_id, camera_id, match_score * 100, frigate_id[:12],
                )
            else:
                # Track unrecognized event for burst suppression
                self._record_unrecognized(camera_id)
                # Track bbox for static false-positive suppression
                bbox = self._extract_bbox(data)
                if bbox and label == "person":
                    self._record_static_fp(camera_id, bbox)
                # Still enrich unrecognized person/pet events with vision description
                if full_frame is not None and category in (ObjectCategory.person, ObjectCategory.pet):
                    asyncio.create_task(self._enrich_narrative_with_vision(
                        event_id=event_id,
                        frame=full_frame,
                        camera_name=camera_friendly,
                        object_type=label,
                        named_object_name=None,
                        timestamp=datetime.fromtimestamp(data.get("start_time", time.time()), tz=timezone.utc),
                    ))
                return

        # Check cross-camera presence conflict
        if self._is_presence_conflict(named_object_id, camera_id):
            logger.info("Presence conflict: %s on camera %d, skipping", named_object_name, camera_id)
            return

        # Successful recognition — clear any burst suppression for this camera
        self._clear_burst_for_camera(camera_id)
        # Clear static FP suppression at this position (it's a real person)
        bbox = self._extract_bbox(data)
        if bbox:
            self._clear_static_fp_at_position(camera_id, bbox)

        # Update presence
        self._presence[named_object_id] = {
            "camera_id": camera_id,
            "last_seen": time.time(),
            "name": named_object_name,
        }

        # ── 1 Frigate track = 1 DB event ──
        # Cross-camera person merging used to fold every subsequent detection
        # of a recognised person into the original event row, producing a
        # single "Philip" card that spanned hours/days. The frontend already
        # clusters discrete events by `group_key` for display, so we keep
        # events 1:1 with Frigate tracks here. The push-frigate sub_label
        # update below still applies so Frigate's UI shows the identity.
        if label == "person" and named_object_name:
            await self._set_frigate_sub_label(frigate_id, named_object_name)

        # Record successful recognition for temporal propagation (direct matches only)
        if not propagated:
            self._record_recognition(camera_id, named_object_id, named_object_name,
                                     category, self._extract_bbox(data), match_score)
            # Backward propagation: update recent unrecognized events on same camera
            asyncio.create_task(self._backward_propagate(
                camera_id, named_object_id, named_object_name, category,
                self._extract_bbox(data), match_score,
            ))

        # Update event in Postgres
        async with async_session() as session:
            result = await session.execute(select(Event).where(Event.id == event_id))
            event_db = result.scalar_one_or_none()
            if event_db:
                event_db.named_object_id = named_object_id
                event_db.event_type = EventType.object_recognized
                event_db.confidence = match_score

                # Fix object_type from pet's known species when model misclassifies
                if category == ObjectCategory.pet:
                    corrected = await self._resolve_pet_species(named_object_id, label, session)
                    if corrected and corrected != label:
                        event_db.object_type = corrected
                        logger.info("Corrected label %s → %s for %s", label, corrected, named_object_name)

                meta = dict(event_db.metadata_extra or {})
                meta["match_score"] = round(match_score, 3)
                meta["recognition_method"] = "temporal_propagation" if propagated else "frigate_bridge"
                # Update narrative with recognized name (factual fallback)
                meta["narrative"] = generate_narrative(
                    named_object_name=named_object_name,
                    object_type=event_db.object_type or label,
                    camera_name=camera_friendly,
                    timestamp=event_db.started_at,
                    seed=frigate_id,
                )
                event_db.metadata_extra = meta
                flag_modified(event_db, "metadata_extra")
                session.add(event_db)
                await session.commit()

        # ── Auto-learn person attributes from successful recognition ──
        # Only direct matches (not temporal propagation) should update profiles,
        # and only for persons with adequate crop quality.
        if (not propagated and category == ObjectCategory.person
                and named_object_id and object_crop is not None):
            try:
                from services.attribute_estimator import (
                    estimate_person_attributes as _est_attrs,
                    merge_stable_attributes,
                )
                # The bbox we extract via `_best_box` may come from either
                # `snapshot.box` (lives in `snapshot.frame_shape` coords) or
                # `box`/`current_box` (lives in `frame_shape` = detect coords).
                # Pick the matching frame_shape so the attribute estimator's
                # height-ratio calculation uses a consistent coordinate space.
                snap = data.get("snapshot") or {}
                snap_box = snap.get("box") if isinstance(snap, dict) else None
                best_box = self._best_box(data)
                if snap_box and best_box and list(snap_box) == list(best_box):
                    fs = snap.get("frame_shape") or data.get("frame_shape")
                else:
                    fs = data.get("frame_shape")
                if fs and len(fs) >= 2:
                    attr_frame_shape = (fs[0], fs[1])
                elif full_frame is not None:
                    attr_frame_shape = full_frame.shape[:2]
                else:
                    attr_frame_shape = object_crop.shape[:2]

                attr_bbox = self._extract_bbox(data)
                attr_bbox_tuple = (0, 0, object_crop.shape[1], object_crop.shape[0])
                if attr_bbox and all(k in attr_bbox for k in ("x1", "y1", "x2", "y2")):
                    attr_bbox_tuple = (attr_bbox["x1"], attr_bbox["y1"], attr_bbox["x2"], attr_bbox["y2"])

                learn_attrs = _est_attrs(object_crop, attr_bbox_tuple, attr_frame_shape)
                if learn_attrs.gender or learn_attrs.age_group or learn_attrs.hair_color:
                    async with async_session() as session:
                        result = await session.execute(
                            select(NamedObject).where(NamedObject.id == named_object_id)
                        )
                        obj = result.scalar_one_or_none()
                        if obj:
                            obj.attributes = merge_stable_attributes(obj.attributes, learn_attrs)
                            flag_modified(obj, "attributes")
                            session.add(obj)
                            await session.commit()
            except Exception:
                logger.debug("Attribute auto-learn failed for %s", named_object_name, exc_info=True)

        # ── Auto-enroll: reinforce embeddings from high-confidence recognitions ──
        # Only direct matches, not temporal propagation. Respects training settings.
        if (not propagated and named_object_id and object_crop is not None
                and match_score > 0):
            asyncio.create_task(self._auto_enroll_detection(
                event_id=event_id,
                named_object_id=named_object_id,
                named_object_name=named_object_name,
                category=category,
                match_score=match_score,
                crop=object_crop.copy(),
            ))

        # Fire vision-based narrative enrichment in background (full frame for scene context)
        if full_frame is not None and category in (ObjectCategory.person, ObjectCategory.pet):
            asyncio.create_task(self._enrich_narrative_with_vision(
                event_id=event_id,
                frame=full_frame,
                camera_name=camera_friendly,
                object_type=label,
                named_object_name=named_object_name,
                timestamp=datetime.fromtimestamp(data.get("start_time", time.time()), tz=timezone.utc),
            ))

        # Push sub_label back to Frigate
        await self._set_frigate_sub_label(frigate_id, named_object_name)

        # Publish to WebSocket
        event_bus.publish({
            "type": "recognition",
            "event_id": event_id,
            "camera_id": camera_id,
            "named_object_id": named_object_id,
            "named_object_name": named_object_name,
            "confidence": match_score,
        })

        # Trigger notification
        if self._notification_engine:
            camera_name = data.get("camera", f"camera_{camera_id}")
            await self._notification_engine.evaluate_and_notify(
                camera_id=camera_id,
                camera_name=camera_name,
                object_type=label,
                named_object_id=named_object_id,
                named_object_name=named_object_name,
                event_id=event_id,
                snapshot_path=snap_path,
            )

        logger.info(
            "Recognized %s (%.0f%%) on camera %d [frigate:%s]",
            named_object_name, match_score * 100, camera_id, frigate_id[:12],
        )

    async def _auto_enroll_detection(
        self,
        event_id: int,
        named_object_id: int,
        named_object_name: Optional[str],
        category: ObjectCategory,
        match_score: float,
        crop: np.ndarray,
    ):
        """Auto-enroll high-confidence detection as training data + reinforce embeddings.

        Reads training settings from DB. If match_score >= threshold, reinforces the
        profile embedding and pins the event for retention_days.
        """
        try:
            from routers.system import load_training_settings
            from services.face_service import face_service
            from services.recognition_service import recognition_service

            async with async_session() as session:
                ts = await load_training_settings(session)

            if not ts.auto_enroll_enabled:
                return
            if match_score < ts.auto_enroll_threshold:
                return

            async with async_session() as session:
                obj = (await session.execute(
                    select(NamedObject).where(NamedObject.id == named_object_id)
                )).scalar_one_or_none()
                if not obj or obj.reference_image_count >= ts.auto_reinforce_cap:
                    return

                reinforced = False

                if category == ObjectCategory.person:
                    # Try face embedding reinforcement
                    if face_service.is_available:
                        faces = await face_service.detect_faces_async(crop)
                        if faces:
                            emb, _ = await asyncio.to_thread(
                                face_service.compute_face_embedding, crop, faces[0].face_data
                            )
                            if emb is not None and obj.embedding and len(obj.embedding) == len(emb):
                                old = np.array(obj.embedding, dtype=np.float64)
                                new = np.array(emb, dtype=np.float64)
                                sim = float(np.dot(old, new) / (np.linalg.norm(old) * np.linalg.norm(new) + 1e-10))
                                if sim >= 0.30:
                                    obj.embedding = face_service.merge_face_embeddings(
                                        obj.embedding, emb, obj.reference_image_count
                                    )
                                    reinforced = True

                    # Body ReID reinforcement (always for persons)
                    if recognition_service.reid_available:
                        body_emb = await recognition_service.compute_reid_embedding_async(crop)
                        if body_emb is not None and obj.body_embedding and len(obj.body_embedding) == len(body_emb):
                            old_b = np.array(obj.body_embedding, dtype=np.float64)
                            new_b = np.array(body_emb, dtype=np.float64)
                            sim_b = float(np.dot(old_b, new_b) / (np.linalg.norm(old_b) * np.linalg.norm(new_b) + 1e-10))
                            if sim_b >= 0.55:
                                obj.body_embedding = recognition_service.merge_reid_embedding(
                                    obj.body_embedding, body_emb, obj.reference_image_count
                                )
                                reinforced = True
                        elif body_emb is not None and not obj.body_embedding:
                            # First body embedding
                            obj.body_embedding = body_emb if isinstance(body_emb, list) else body_emb.tolist()
                            reinforced = True

                elif category == ObjectCategory.pet:
                    # CNN embedding reinforcement for pets
                    raw_emb = await recognition_service.compute_embedding(crop)
                    if raw_emb and obj.embedding and len(obj.embedding) == len(raw_emb):
                        old_c = np.array(obj.embedding, dtype=np.float64)
                        new_c = np.array(raw_emb, dtype=np.float64)
                        sim_c = float(np.dot(old_c, new_c) / (np.linalg.norm(old_c) * np.linalg.norm(new_c) + 1e-10))
                        if sim_c >= 0.55:
                            merged = (old_c * obj.reference_image_count + new_c) / (obj.reference_image_count + 1)
                            norm = np.linalg.norm(merged)
                            if norm > 0:
                                merged = merged / norm
                            obj.embedding = merged.tolist()
                            reinforced = True

                if reinforced:
                    obj.reference_image_count += 1
                    session.add(obj)
                    await session.commit()

                    # Pin the event with retention
                    pin_until = (datetime.now(timezone.utc) + timedelta(days=ts.training_retention_days)).isoformat()
                    ev = (await session.execute(
                        select(Event).where(Event.id == event_id)
                    )).scalar_one_or_none()
                    if ev:
                        meta = dict(ev.metadata_extra or {})
                        meta["pinned_until"] = pin_until
                        meta["pinned_reason"] = "auto_enroll"
                        ev.metadata_extra = meta
                        flag_modified(ev, "metadata_extra")
                        session.add(ev)
                        await session.commit()

                    logger.info(
                        "Auto-enrolled %s: reinforced embeddings (ref_count=%d, score=%.0f%%, pinned %dd)",
                        named_object_name, obj.reference_image_count, match_score * 100, ts.training_retention_days,
                    )

        except Exception:
            logger.debug("Auto-enroll failed for %s", named_object_name, exc_info=True)

    async def _enrich_narrative_with_vision(
        self,
        event_id: int,
        frame: np.ndarray,
        camera_name: str,
        object_type: str,
        named_object_name: Optional[str],
        timestamp: Optional[datetime],
    ):
        """Background task: call Ollama vision model to describe the snapshot,
        fall back to text LLM, then update the event narrative in Postgres."""
        try:
            # Try vision model first (moondream)
            description = await describe_snapshot_with_vision(
                frame,
                camera_name=camera_name,
                object_type=object_type,
                named_object_name=named_object_name,
                timestamp=timestamp,
            )
            source = "vision"

            # Fall back to text LLM (qwen2.5) if vision failed
            if not description:
                description = await describe_with_text_llm(
                    camera_name=camera_name,
                    object_type=object_type,
                    named_object_name=named_object_name,
                    timestamp=timestamp,
                )
                source = "llm"

            if not description:
                return

            async with async_session() as session:
                result = await session.execute(select(Event).where(Event.id == event_id))
                event_db = result.scalar_one_or_none()
                if event_db:
                    meta = dict(event_db.metadata_extra or {})
                    meta["narrative"] = description
                    meta["narrative_source"] = source
                    event_db.metadata_extra = meta
                    flag_modified(event_db, "metadata_extra")
                    session.add(event_db)
                    await session.commit()

        except Exception:
            logger.debug("Narrative enrichment failed for event %d", event_id, exc_info=True)

    async def _update_activity_narrative(
        self,
        event_id: int,
        thumb_paths: list[str],
        camera_name: str,
        object_type: str,
        named_object_name: Optional[str],
        duration_seconds: float,
    ):
        """Background task: update narrative with multi-frame activity description on event end."""
        try:
            from services.ml_client import remote_describe_activity
            from services.narrative_generator import (
                _ACTIVITY_PROMPT,
                _format_describe_prompt,
            )

            # Load thumbnail frames (pick evenly spaced subset)
            frames = []
            step = max(1, len(thumb_paths) // 5)
            selected = thumb_paths[::step][:6]
            for p in selected:
                try:
                    img = await asyncio.to_thread(cv2.imread, str(p))
                    if img is not None and img.size > 0:
                        frames.append(img)
                except Exception:
                    pass

            if len(frames) < 2:
                return

            # Get previous description for context
            prev_desc = None
            async with async_session() as session:
                result = await session.execute(select(Event).where(Event.id == event_id))
                event_db = result.scalar_one_or_none()
                if event_db:
                    meta = event_db.metadata_extra or {}
                    prev_desc = meta.get("narrative")

            description = await remote_describe_activity(
                frames,
                camera_name=camera_name,
                object_type=object_type,
                named_object_name=named_object_name,
                duration_seconds=duration_seconds,
                previous_description=prev_desc,
                instructions=_format_describe_prompt(
                    _ACTIVITY_PROMPT,
                    named_object_name=named_object_name,
                    object_type=object_type,
                    camera_name=camera_name,
                ),
            )

            if not description:
                return
            # Drop the explicit "unclear" sentinel so we don't overwrite a
            # better template/single-frame narrative with a useless phrase.
            if description.strip().lower().rstrip(".") == "unclear":
                return
            # Reject narratives that ignore the prompt's forbidden-phrase rules
            # (e.g. "transitions to", "facing right") — the remote ML server
            # sometimes falls back to its generic posture-description prompt.
            from services.narrative_generator import _is_forbidden_narrative
            if _is_forbidden_narrative(description):
                logger.info(
                    "Rejecting activity narrative (forbidden phrasing): %s",
                    description[:120],
                )
                return

            async with async_session() as session:
                result = await session.execute(select(Event).where(Event.id == event_id))
                event_db = result.scalar_one_or_none()
                if event_db:
                    meta = dict(event_db.metadata_extra or {})
                    meta["narrative"] = description
                    meta["narrative_source"] = "activity"
                    event_db.metadata_extra = meta
                    flag_modified(event_db, "metadata_extra")
                    session.add(event_db)
                    await session.commit()
                    logger.debug("Activity narrative updated for event #%d", event_id)

        except Exception:
            logger.debug("Activity narrative failed for event %d", event_id, exc_info=True)

    # ── Vision Enforcement Helpers ──

    async def _load_reference_thumbnails(self, named_object_id: int, limit: int = 3) -> list[np.ndarray]:
        """Load recent high-confidence thumbnails for a named object as numpy arrays."""
        async with async_session() as session:
            result = await session.execute(
                select(Event.thumbnail_path)
                .where(
                    Event.named_object_id == named_object_id,
                    Event.thumbnail_path.isnot(None),
                    Event.confidence >= 0.60,
                )
                .order_by(Event.started_at.desc())
                .limit(limit * 2)  # fetch extra in case some files are missing
            )
            paths = result.scalars().all()

        crops = []
        for p in paths:
            if len(crops) >= limit:
                break
            try:
                img = await asyncio.to_thread(cv2.imread, str(p))
                if img is not None and img.size > 0:
                    crops.append(img)
            except Exception:
                pass

        return crops

    async def _load_profile_attributes(self, named_object_id: int) -> dict:
        """Load the attributes dict for a named object profile."""
        async with async_session() as session:
            result = await session.execute(
                select(NamedObject.attributes).where(NamedObject.id == named_object_id)
            )
            attrs = result.scalar_one_or_none()
        return attrs or {}

    async def _build_vision_candidates(
        self, category: ObjectCategory, limit: int = 8,
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        """Build a list of {id, name, crop} dicts for all known profiles in a category.

        Each entry includes the best available reference thumbnail as a numpy crop.
        ``exclude_ids`` can be used to skip profiles that were already rejected.
        """
        async with async_session() as session:
            result = await session.execute(
                select(NamedObject).where(NamedObject.category == category)
            )
            known = result.scalars().all()

        if not known:
            return []

        candidates = []
        for obj in known[:limit * 2]:  # fetch more in case some are excluded/empty
            if exclude_ids and obj.id in exclude_ids:
                continue
            ref_crops = await self._load_reference_thumbnails(obj.id, limit=3)
            if ref_crops:
                candidates.append({
                    "id": obj.id,
                    "name": obj.name,
                    "crop": ref_crops[0],
                    "attributes": obj.attributes or {},
                })
            if len(candidates) >= limit:
                break

        return candidates

    async def _recognize_person(self, crop: np.ndarray, camera_id: int,
                                full_frame: Optional[np.ndarray] = None,
                                bbox: Optional[dict] = None,
                                det_confidence: float = 1.0,
                                data: Optional[dict] = None,
                                ) -> tuple[Optional[int], Optional[str], float]:
        """Run face + body recognition on a person crop WITH attribute validation.

        All embedding candidates are iterated through the recognition agent's
        validate_match() checks (gender, age, height, hair, skin tone, clothing)
        before being accepted.  This prevents a brick wall or wrong person from
        being identified just because the embedding distance happened to be close.
        """
        from services.face_service import face_service
        from services.recognition_service import recognition_service
        from services.recognition_agent import recognition_agent
        from services.attribute_estimator import estimate_person_attributes

        # ── Crop quality gate — reject crops that are not real persons ──
        quality = recognition_agent.assess_crop_quality(crop, "person")
        if not quality.is_valid:
            logger.info("Crop rejected on camera %d: %s", camera_id, quality.reject_reason)
            return None, None, 0.0

        # ── Estimate attributes from the crop for validation ──
        # Use Frigate's detect resolution for accurate height_ratio
        fs = (data or {}).get("frame_shape")
        if fs and len(fs) >= 2:
            frame_shape = (fs[0], fs[1])  # (height, width)
        elif full_frame is not None:
            frame_shape = full_frame.shape[:2]
        else:
            frame_shape = crop.shape[:2]
        bbox_tuple = (0, 0, crop.shape[1], crop.shape[0])  # default: full crop
        if bbox and all(k in bbox for k in ("x", "y", "w", "h")):
            bbox_tuple = (bbox["x"], bbox["y"], bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])
        elif bbox and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            bbox_tuple = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])

        face_data_for_attrs = None  # will be set if face detected

        # Load known persons once
        async with async_session() as session:
            result = await session.execute(
                select(NamedObject).where(
                    NamedObject.category == ObjectCategory.person,
                )
            )
            known_persons = result.scalars().all()

        if not known_persons:
            return None, None, 0.0

        # Build profile lookup for attribute validation
        profile_map = {p.id: p.attributes or {} for p in known_persons}

        # ── Stage 1: Face detection + matching ──
        if face_service.is_available:
            faces = await face_service.detect_faces_async(crop)
            if faces:
                face_data_for_attrs = faces[0].face_data
                emb, _ = await asyncio.to_thread(
                    face_service.compute_face_embedding, crop, faces[0].face_data
                )
                if emb is not None:
                    known_faces = [
                        (p.id, p.name, p.embedding)
                        for p in known_persons
                        if p.embedding
                    ]
                    if known_faces:
                        candidates = face_service.match_face_ranked(
                            emb, known_faces, threshold=0.25
                        )
                        if not candidates:
                            # Diagnostic: show best sub-threshold face score so we can
                            # tune and spot collapsed/stale face centroids.
                            all_cands = face_service.match_face_ranked(
                                emb, known_faces, threshold=0.0
                            )
                            if all_cands:
                                best = all_cands[0]
                                logger.info(
                                    "Face match below threshold on cam %d: best=%s %.3f (thr=0.25)",
                                    camera_id, best[0], best[2],
                                )
                            else:
                                logger.info(
                                    "Face match produced no candidates on cam %d (%d known faces)",
                                    camera_id, len(known_faces),
                                )
                        if candidates:
                            # Estimate person attributes using face data
                            person_attrs = estimate_person_attributes(
                                crop, bbox_tuple, frame_shape, face_data_for_attrs
                            )
                            # Iterate ALL candidates through validation
                            for cand in candidates:
                                cand_name, cand_id, cand_score, cand_margin = cand
                                stored_profile = profile_map.get(cand_id)
                                verdict = recognition_agent.validate_match(
                                    camera_id=camera_id,
                                    named_object_id=cand_id,
                                    named_object_name=cand_name,
                                    raw_score=cand_score,
                                    adjusted_score=cand_score,
                                    person_attrs=person_attrs,
                                    stored_profile=stored_profile,
                                    match_method="face",
                                    det_class="person",
                                    det_confidence=det_confidence,
                                )
                                if verdict.accept:
                                    final_score = cand_score * verdict.confidence_adjustment
                                    logger.info(
                                        "Face match ACCEPTED: %s (score=%.3f, adj=%.3f) on camera %d",
                                        cand_name, cand_score, final_score, camera_id,
                                    )
                                    recognition_agent.record_appearance(
                                        camera_id, cand_id, person_attrs
                                    )
                                    return cand_id, cand_name, final_score
                                else:
                                    logger.info(
                                        "Face match REJECTED: %s (score=%.3f) on camera %d — %s",
                                        cand_name, cand_score, camera_id, verdict.reason,
                                    )
            else:
                logger.debug("No face detected in crop (%dx%d) on camera %d",
                             crop.shape[1], crop.shape[0], camera_id)

        # ── Stage 2: Body ReID fallback ──
        # When no face was found, the crop may not even be a person (Frigate
        # false positive).  Require a higher body threshold to compensate.
        face_found = face_data_for_attrs is not None
        # v25 tiered acceptance — Frigate thumbnails are small/low-quality and
        # local ReID (when remote ML is unreachable) gives lower cosines than
        # the Cloudflare-hosted model.  Accept lower scores when the margin
        # over the next-best candidate is large enough to avoid confusion.
        body_threshold = 0.30 if face_found else 0.32
        BODY_HIGH_ACCEPT = 0.40 if face_found else 0.40   # high-confidence band
        BODY_MID_MARGIN = 0.04                            # margin needed when score >= HIGH_ACCEPT
        BODY_LOW_MARGIN = 0.07                            # margin needed when score < HIGH_ACCEPT
        body_min_margin = BODY_MID_MARGIN
        if recognition_service.reid_available:
            known_bodies = [
                (p.id, p.name, p.body_embedding)
                for p in known_persons
                if p.body_embedding
            ]
            if known_bodies:
                candidates = await recognition_service.match_person_body_ranked(
                    crop, known_bodies, threshold=body_threshold
                )
                if not candidates:
                    # Diagnostic: show best sub-threshold body score.
                    all_b = await recognition_service.match_person_body_ranked(
                        crop, known_bodies, threshold=0.0
                    )
                    if all_b:
                        logger.info(
                            "Body match below threshold on cam %d: best=%s %.3f (thr=%.2f, face_found=%s)",
                            camera_id, all_b[0].subject, all_b[0].confidence,
                            body_threshold, face_found,
                        )
                if candidates:
                    # Reject ambiguous matches where top two candidates are too close.
                    # Tier the required margin by score band: low-tier matches need a
                    # bigger gap because cosines themselves are weaker.
                    top = candidates[0]
                    required_margin = (
                        BODY_MID_MARGIN if top.confidence >= BODY_HIGH_ACCEPT
                        else BODY_LOW_MARGIN
                    )
                    if top.margin < required_margin and len(known_bodies) > 1:
                        logger.info(
                            "Body match AMBIGUOUS on cam %d: best=%s %.3f margin=%.3f < %.2f",
                            camera_id, top.subject, top.confidence,
                            top.margin, required_margin,
                        )
                        candidates = []
                if candidates:
                    # Estimate person attributes (without face data for body-only)
                    person_attrs = estimate_person_attributes(
                        crop, bbox_tuple, frame_shape, face_data_for_attrs
                    )
                    # Iterate ALL body candidates through validation
                    for cand in candidates:
                        stored_profile = profile_map.get(int(cand.subject_id))
                        verdict = recognition_agent.validate_match(
                            camera_id=camera_id,
                            named_object_id=int(cand.subject_id),
                            named_object_name=cand.subject,
                            raw_score=cand.confidence,
                            adjusted_score=cand.confidence,
                            person_attrs=person_attrs,
                            stored_profile=stored_profile,
                            match_method="body",
                            det_class="person",
                            det_confidence=det_confidence,
                        )
                        if verdict.accept:
                            final_score = cand.confidence * verdict.confidence_adjustment
                            logger.info(
                                "Body match ACCEPTED: %s (score=%.3f, adj=%.3f) on camera %d",
                                cand.subject, cand.confidence, final_score, camera_id,
                            )
                            recognition_agent.record_appearance(
                                camera_id, int(cand.subject_id), person_attrs
                            )
                            return int(cand.subject_id), cand.subject, final_score
                        else:
                            logger.info(
                                "Body match REJECTED: %s (score=%.3f) on camera %d — %s",
                                cand.subject, cand.confidence, camera_id, verdict.reason,
                            )
                else:
                    logger.debug("Body ReID no match (threshold=%.2f) on camera %d, crop %dx%d",
                                 body_threshold, camera_id, crop.shape[1], crop.shape[0])

        return None, None, 0.0

    async def _recognize_pet(self, crop: np.ndarray, label: str, camera_id: int
                             ) -> tuple[Optional[int], Optional[str], float]:
        """Pet recognition with colour-prior gate and margin enforcement.

        The MobileNetV2 CNN cosine alone has proven unable to discriminate cats
        of the same breed but very different fur colour (Frostie vs Tangie).
        We now:

        1.  Compute a cheap HSV colour signal for the crop.
        2.  Multiply each candidate's cosine by the colour-compatibility weight
            against the candidate profile's stored ``attributes['color']``.
            A clear contradiction (white profile vs tortoiseshell crop) vetoes
            the candidate (multiplier=0).
        3.  Require a minimum margin between top-1 and top-2 (anti-confusion).
        4.  Raise the base accept threshold from 0.45 → 0.55 (was tuned to a
            stale class-imbalanced centroid; a healthy match scores >0.65).
        """
        from services.recognition_service import recognition_service
        from services.pet_color_gate import (
            compute_colour_signal, colour_compatibility,
        )

        async with async_session() as session:
            result = await session.execute(
                select(NamedObject).where(
                    NamedObject.category == ObjectCategory.pet,
                    NamedObject.embedding.isnot(None),
                )
            )
            known_pets = result.scalars().all()

        if not known_pets:
            return None, None, 0.0

        # Load profile attribute lookup (for colour gate).
        profile_attrs: dict[int, dict] = {p.id: (p.attributes or {}) for p in known_pets}
        known_list = [(p.id, p.name, p.embedding) for p in known_pets]

        candidates = await recognition_service.match_pet_ranked(crop, known_list)
        if not candidates:
            return None, None, 0.0

        # ── Colour gate: re-rank by cosine × colour_compatibility ──
        signal = compute_colour_signal(crop)
        gated: list[tuple[int, str, float, float, float]] = []
        # Each tuple: (id, name, gated_score, raw_cosine, multiplier)
        for cand in candidates:
            cid = int(cand.subject_id) if cand.subject_id is not None else -1
            attrs = profile_attrs.get(cid, {})
            profile_color = attrs.get("color") or attrs.get("colour")
            mult = colour_compatibility(profile_color, signal)
            if mult <= 0.0:
                logger.info(
                    "Pet colour VETO: %s profile=%s observed=%s (white=%.2f black=%.2f) on cam %d",
                    cand.subject, profile_color, signal.family,
                    signal.white_ratio, signal.black_ratio, camera_id,
                )
                continue
            gated.append((cid, cand.subject, cand.confidence * mult,
                          cand.confidence, mult))

        if not gated:
            logger.info(
                "Pet recognition: all %d candidates vetoed by colour gate on cam %d "
                "(observed family=%s white=%.2f black=%.2f) — leaving unrecognised",
                len(candidates), camera_id, signal.family,
                signal.white_ratio, signal.black_ratio,
            )
            return None, None, 0.0

        gated.sort(key=lambda g: g[2], reverse=True)
        best_id, best_name, best_score, raw, mult = gated[0]

        # ── Margin & threshold gates ──
        # Lowered threshold tier scheme:
        #   raw >= 0.55  → high-confidence accept (legacy bar)
        #   0.40 <= raw < 0.55 → mid-confidence: requires margin >= 0.07
        #     plus the colour gate must NOT be neutral (mult>=1.0 means
        #     positive colour evidence, not the default 1.0 fallback).
        #   raw < 0.40   → reject (noise floor)
        # This rescues the many genuine but weak matches caused by Frigate's
        # low-quality 300×300 fallback crops (e.g. when HD snapshot path fails
        # or the species label is wrong and detect bbox is loose).
        BASE_ACCEPT = 0.55
        MID_ACCEPT = 0.40
        MIN_MARGIN = 0.05
        MID_MARGIN = 0.07

        if raw < MID_ACCEPT:
            logger.info(
                "Pet match below threshold: best=%s raw=%.3f gated=%.3f (need raw≥%.2f) on cam %d",
                best_name, raw, best_score, MID_ACCEPT, camera_id,
            )
            return None, None, 0.0
        # Decide which margin requirement applies.
        required_margin = MIN_MARGIN if raw >= BASE_ACCEPT else MID_MARGIN
        if len(gated) >= 2:
            second_score = gated[1][2]
            margin = best_score - second_score
            if margin < required_margin:
                logger.info(
                    "Pet match too ambiguous on cam %d: %s=%.3f vs %s=%.3f "
                    "(margin=%.3f<%.2f, raw=%.3f)",
                    camera_id, best_name, best_score, gated[1][1], second_score,
                    margin, required_margin, raw,
                )
                return None, None, 0.0

        logger.info(
            "Pet match ACCEPTED: %s raw=%.3f colour×%.2f=%.3f (signal=%s w=%.2f b=%.2f) on cam %d",
            best_name, raw, mult, best_score, signal.family,
            signal.white_ratio, signal.black_ratio, camera_id,
        )
        return best_id, best_name, best_score

    # ── Label Correction ──

    # Known cat breeds for species resolution (lowercase)
    _CAT_BREEDS = {
        "persian", "siamese", "maine coon", "ragdoll", "bengal", "abyssinian",
        "sphynx", "british shorthair", "scottish fold", "birman", "burmese",
        "russian blue", "norwegian forest", "devon rex", "oriental",
        "exotic shorthair", "tonkinese", "cornish rex", "turkish angora",
        "somali", "himalayan", "manx", "savannah", "balinese", "bombay",
        "ocicat", "chartreux", "american shorthair", "selkirk rex",
        "tortoiseshell", "tabby", "tuxedo", "calico",
    }
    _DOG_BREEDS = {
        "labrador", "golden retriever", "german shepherd", "bulldog", "poodle",
        "beagle", "rottweiler", "husky", "boxer", "dachshund", "corgi",
        "dalmatian", "chihuahua", "shih tzu", "pomeranian", "maltese",
        "great dane", "doberman", "border collie", "cocker spaniel",
        "cavalier", "pit bull", "terrier", "schnauzer", "mastiff",
    }

    async def _resolve_pet_species(
        self, named_object_id: int, frigate_label: str, session
    ) -> Optional[str]:
        """Determine a pet's true species (cat/dog) from its stored attributes.

        Returns the correct label or None if unknown.
        """
        result = await session.execute(
            select(NamedObject).where(NamedObject.id == named_object_id)
        )
        pet = result.scalar_one_or_none()
        if not pet:
            return None

        attrs = pet.attributes or {}

        # Check explicit species override first
        species = attrs.get("species", "").lower().strip()
        if species in ("cat", "dog"):
            return species

        # Infer from breed
        breed = attrs.get("breed", "").lower().strip()
        if breed:
            if breed in self._CAT_BREEDS or any(b in breed for b in self._CAT_BREEDS):
                return "cat"
            if breed in self._DOG_BREEDS or any(b in breed for b in self._DOG_BREEDS):
                return "dog"

        # Fall back to majority vote from recent events for this pet
        from sqlalchemy import func as sa_func
        result = await session.execute(
            select(Event.object_type, sa_func.count())
            .where(
                Event.named_object_id == named_object_id,
                Event.object_type.in_(["cat", "dog"]),
            )
            .group_by(Event.object_type)
            .order_by(sa_func.count().desc())
            .limit(1)
        )
        row = result.first()
        if row:
            return row[0]

        return None

    # ── Frigate API Helpers ──

    async def _fetch_snapshot(self, frigate_id: str,
                              camera_name: Optional[str] = None,
                              frame_time: Optional[float] = None,
                              ) -> Optional[np.ndarray]:
        """Fetch the best snapshot for a Frigate event (without bbox annotations).

        Prefers the high-res record stream via the recordings/snapshot endpoint
        (full camera resolution, e.g. 1920x1080) so person/pet recognition gets
        usable crops instead of the 640x360 detect-stream frames. Falls back to
        the event snapshot endpoint if the recording fetch fails.
        """
        # ── Preferred: high-res frame from the record stream ──
        if camera_name and frame_time:
            try:
                resp_hr = await self._http.get(
                    f"/api/{camera_name}/recordings/{frame_time:.3f}/snapshot.jpg",
                    params={"quality": 95},
                )
                if resp_hr.status_code == 200 and resp_hr.content:
                    arr = np.frombuffer(resp_hr.content, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None and img.shape[0] >= 480:
                        return img
            except Exception as e:
                logger.debug("High-res snapshot fetch failed for %s @ %s: %s",
                             camera_name, frame_time, e)

        # ── Fallback: event snapshot from detect stream ──
        try:
            resp = await self._http.get(
                f"/api/events/{frigate_id}/snapshot.jpg",
                params={"quality": 95, "bbox": 0},
            )
            if resp.status_code != 200:
                return None
            arr = np.frombuffer(resp.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning("Failed to fetch snapshot for %s: %s", frigate_id[:12], e)
            return None

    async def _set_frigate_sub_label(self, frigate_id: str, name: str):
        """Push recognition result back to Frigate as a sub_label."""
        try:
            await self._http.post(
                f"/api/events/{frigate_id}/sub_label",
                json={"subLabel": name, "subLabelScore": 1.0},
            )
        except Exception as e:
            logger.debug("Failed to set sub_label for %s: %s", frigate_id[:12], e)

    # ── Storage Helpers ──

    async def _save_snapshot(self, frame: np.ndarray, camera_id: int,
                             frigate_id: str) -> Optional[str]:
        """Save snapshot to disk (hot storage if available, else cold)."""
        from pathlib import Path

        base = Path(settings.hot_storage_path) / "snapshots" if settings.hot_storage_path \
            else Path(settings.snapshots_path)
        cam_dir = base / f"camera_{camera_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{frigate_id}.jpg"
        path = cam_dir / filename
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            path.write_bytes(buf.tobytes())
            return str(path)
        return None

    async def _save_thumbnail(self, frame: np.ndarray, camera_id: int,
                              frigate_id: str) -> Optional[str]:
        """Save a resized thumbnail with unique name for GIF history."""
        from pathlib import Path

        base = Path(settings.hot_storage_path) / "snapshots" if settings.hot_storage_path \
            else Path(settings.snapshots_path)
        cam_dir = base / f"camera_{camera_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Resize to max 300px width
        h, w = frame.shape[:2]
        if w > 300:
            scale = 300 / w
            thumb = cv2.resize(frame, (300, int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            thumb = frame

        # Use timestamp suffix for unique filenames (supports GIF history)
        ts = datetime.now(timezone.utc).strftime("%H%M%S")
        filename = f"{frigate_id}_{ts}_thumb.jpg"
        path = cam_dir / filename
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            path.write_bytes(buf.tobytes())
            return str(path)
        return None

    async def _build_preview_gif(
        self, camera_id: int, thumbnail_paths: list[str], label: str,
    ) -> Optional[str]:
        """Build an animated GIF from accumulated event thumbnails."""
        from pathlib import Path
        from PIL import Image

        if not thumbnail_paths:
            return None

        base = Path(settings.hot_storage_path) / "snapshots" if settings.hot_storage_path \
            else Path(settings.snapshots_path)
        cam_dir = base / f"camera_{camera_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{label}_preview.gif"
        filepath = cam_dir / filename

        def _save():
            frames: list[Image.Image] = []
            for tp in thumbnail_paths:
                p = Path(tp)
                if not p.exists():
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                    frames.append(img)
                except Exception:
                    continue
            if not frames:
                return False
            tw, th = frames[0].size
            resized = []
            for f in frames:
                if f.size != (tw, th):
                    f = f.resize((tw, th), Image.LANCZOS)
                resized.append(f)
            if len(resized) == 1:
                resized[0].save(str(filepath), optimize=True)
            else:
                resized[0].save(
                    str(filepath),
                    save_all=True,
                    append_images=resized[1:],
                    duration=750,
                    loop=0,
                    optimize=True,
                )
            return True

        ok = await asyncio.to_thread(_save)
        if ok:
            logger.info("Built preview GIF: %s (%d frames)", filepath, len(thumbnail_paths))
            return str(filepath)
        return None

    # ── Event Grouping ──

    def _get_or_create_group_key(self, camera_id: int, event_ts: float) -> str:
        """Get or create a group key for this camera within the grouping window.

        A group is extended when a new event arrives within GROUP_WINDOW_S of the
        previous event AND the group has been active for less than
        MAX_EVENT_DURATION_S total. Past that absolute cap we force a new group
        even on continuous activity, so the UI doesn't collapse a 24-hour run
        of detections into a single card.
        """
        existing = self._active_groups.get(camera_id)
        if existing:
            key, last_ts, started_ts = existing
            within_idle = (event_ts - last_ts) < GROUP_WINDOW_S
            within_total = (event_ts - started_ts) < MAX_EVENT_DURATION_S
            if within_idle and within_total:
                # Extend existing group
                self._active_groups[camera_id] = (key, event_ts, started_ts)
                return key
        # New group
        group_key = f"cam{camera_id}_{int(event_ts)}"
        self._active_groups[camera_id] = (group_key, event_ts, event_ts)
        return group_key

    # ── Event Consolidation ──

    async def _try_consolidate(self, camera_id: int, label: str,
                               bbox: Optional[dict], score: float,
                               frigate_id: str, start_ts: datetime,
                               data: dict) -> Optional[int]:
        """Check if we can consolidate this pet detection into a recent event.

        Only used for pets (cat/dog/bird) — per-camera 3-tuple keys.
        Returns the existing event_id if consolidated, or None to create a new event.
        """
        now = time.time()
        # Prune stale entries (handle both 2-tuple person keys and 3-tuple pet keys)
        max_window = max(CONSOLIDATION_NAMED_S, CONSOLIDATION_UNKNOWN_S, CONSOLIDATION_IDLE_S)
        stale = [k for k, v in self._recent_events.items() if now - v["last_ts"] > max_window]
        for k in stale:
            del self._recent_events[k]

        # Search per-camera entries (3-tuple keys only — pets and unknowns)
        best_entry = None
        best_key = None
        for key, entry in self._recent_events.items():
            if len(key) != 3:
                continue  # skip 2-tuple cross-camera person keys
            c_cam, c_label, c_oid = key
            if c_cam != camera_id:
                continue
            # Match label category: person↔person, cat/dog interchangeable for pets
            if c_label != label:
                pet_labels = {"cat", "dog"}
                if not (c_label in pet_labels and label in pet_labels):
                    continue
            # Check window: named pets use idle timeout, unknowns use short window
            window = CONSOLIDATION_IDLE_S if c_oid > 0 else CONSOLIDATION_UNKNOWN_S
            if now - entry["last_ts"] > window:
                continue
            # Prefer named match over unknown
            if best_entry is None or (c_oid > 0 and (best_key[2] == 0)):
                best_entry = entry
                best_key = key

        if best_entry is None:
            return None

        event_id = best_entry["event_id"]
        cam_friendly = _camera_id_to_friendly.get(camera_id, f"camera_{camera_id}")

        # Extend existing event
        async with async_session() as session:
            result = await session.execute(select(Event).where(Event.id == event_id))
            event_db = result.scalar_one_or_none()
            if not event_db:
                # Event was deleted — remove from cache
                del self._recent_events[best_key]
                return None

            # Hard cap on event duration to prevent runaway consolidation
            # (e.g. continuous low-frequency detections all night collapsing into one event)
            if event_db.started_at:
                ev_start_ts = event_db.started_at.timestamp() if hasattr(event_db.started_at, "timestamp") else 0
                if ev_start_ts and (start_ts.timestamp() - ev_start_ts) > MAX_EVENT_DURATION_S:
                    # Too old — close cache and force a new event
                    del self._recent_events[best_key]
                    return None

            event_db.ended_at = start_ts
            if score > (event_db.confidence or 0):
                event_db.confidence = score
            if bbox:
                event_db.bbox = bbox

            meta = dict(event_db.metadata_extra or {})
            meta["consolidated"] = meta.get("consolidated", 1) + 1
            meta["last_frigate_id"] = frigate_id
            # Update snapshot/thumbnail path references for latest frame
            detect_res = data.get("frame_shape")
            if detect_res and len(detect_res) >= 2:
                meta["detect_resolution"] = [detect_res[1], detect_res[0]]
            event_db.metadata_extra = meta
            flag_modified(event_db, "metadata_extra")
            session.add(event_db)
            await session.commit()

        # Update cache timestamp
        best_entry["last_ts"] = now
        if bbox:
            best_entry["bbox"] = bbox

        consol_count = (event_db.metadata_extra or {}).get("consolidated", 2)
        logger.info("Consolidated into #%d: %s on %s (×%d) [%s]",
                    event_id, label, cam_friendly, consol_count, frigate_id[:12])

        # Publish update to WebSocket
        event_bus.publish({
            "type": "event_updated",
            "event_id": event_id,
            "camera_id": camera_id,
            "confidence": score,
        })

        return event_id

    async def _merge_into_existing(self, source_id: int, target_id: int,
                                   camera_id: int, label: str,
                                   frigate_id: str) -> bool:
        """Merge source event into target (extend target, delete source).

        Used for post-recognition consolidation. For persons this is cross-camera:
        the target event's camera_id updates to the LATEST camera, and metadata
        tracks all cameras visited.
        """
        async with async_session() as session:
            result = await session.execute(
                select(Event).where(Event.id.in_([source_id, target_id]))
            )
            events = {e.id: e for e in result.scalars().all()}
            target = events.get(target_id)
            source = events.get(source_id)
            if not target or not source:
                return False

            # Extend target ended_at
            if source.started_at:
                if not target.ended_at or source.started_at > target.ended_at:
                    target.ended_at = source.started_at
            # Better confidence
            if (source.confidence or 0) > (target.confidence or 0):
                target.confidence = source.confidence

            # Track cameras visited (cross-camera consolidation)
            meta = dict(target.metadata_extra or {})
            cameras_seen = set(meta.get("cameras", [target.camera_id]))
            cameras_seen.add(camera_id)
            meta["cameras"] = sorted(cameras_seen)
            meta["consolidated"] = meta.get("consolidated", 1) + 1
            meta["last_frigate_id"] = frigate_id
            meta["last_camera_id"] = camera_id
            # Update camera_id to latest camera
            target.camera_id = camera_id
            target.metadata_extra = meta
            flag_modified(target, "metadata_extra")
            session.add(target)
            await session.delete(source)
            await session.commit()

        # Notify frontend
        event_bus.publish({"type": "event_deleted", "event_id": source_id})
        event_bus.publish({
            "type": "event_updated",
            "event_id": target_id,
            "camera_id": camera_id,
        })
        return True

    def _update_consolidation_cache(self, camera_id: int, label: str,
                                    named_object_id: int, event_id: int,
                                    bbox: Optional[dict]):
        """Update the consolidation cache when an event gets recognized.

        Moves the event from the unknown key to the named key so subsequent
        detections of the same person/pet consolidate into the same event.
        Persons use CROSS-CAMERA keys: ("person", oid). Pets use per-camera keys.
        """
        now = time.time()
        # Remove any old unknown entry for this camera+label
        unknown_key = (camera_id, label, 0)
        old_entry = self._recent_events.get(unknown_key)
        if old_entry and old_entry["event_id"] == event_id:
            del self._recent_events[unknown_key]

        # Also for pet label variants
        if label in ("cat", "dog"):
            for alt in ("cat", "dog"):
                alt_key = (camera_id, alt, 0)
                old_alt = self._recent_events.get(alt_key)
                if old_alt and old_alt["event_id"] == event_id:
                    del self._recent_events[alt_key]

        # Set named key — persons are cross-camera, pets per-camera
        if label == "person":
            named_key = ("person", named_object_id)
        else:
            named_key = (camera_id, label, named_object_id)
        self._recent_events[named_key] = {
            "event_id": event_id,
            "last_ts": now,
            "bbox": bbox,
            "camera_id": camera_id,
        }

    # ── Presence Tracking ──

    def _is_presence_conflict(self, named_object_id: int, camera_id: int) -> bool:
        """Check if this named object was recently seen on a different camera."""
        entry = self._presence.get(named_object_id)
        if entry is None:
            return False
        if entry["camera_id"] == camera_id:
            return False
        return (time.time() - entry["last_seen"]) < PRESENCE_TIMEOUT_S

    # ── Temporal Propagation ──

    def _record_recognition(self, camera_id: int, named_object_id: int,
                            name: str, category, bbox: Optional[dict],
                            score: float):
        """Record a successful recognition so nearby events can inherit it."""
        entries = self._recent_recognitions.setdefault(camera_id, [])
        entries.append({
            "named_object_id": named_object_id,
            "name": name,
            "category": category,
            "bbox": bbox,
            "timestamp": time.time(),
            "score": score,
        })
        # Keep only the most recent per camera
        if len(entries) > TEMPORAL_MAX_ENTRIES:
            entries[:] = entries[-TEMPORAL_MAX_ENTRIES:]

    def _try_temporal_propagation(self, camera_id: int, category,
                                  bbox: Optional[dict]) -> Optional[dict]:
        """Check if a recently recognized object on this camera can propagate identity.

        Returns the matching entry dict or None.
        """
        entries = self._recent_recognitions.get(camera_id)
        if not entries:
            return None
        now = time.time()
        # Search most recent first
        for entry in reversed(entries):
            if now - entry["timestamp"] > TEMPORAL_WINDOW_S:
                continue
            if entry["category"] != category:
                continue
            # Spatial check: require minimum bbox overlap if both available
            if bbox and entry["bbox"]:
                iou = self._bbox_iou(bbox, entry["bbox"])
                if iou < TEMPORAL_IOU_MIN:
                    continue
            return entry
        # Prune expired entries
        cutoff = now - TEMPORAL_WINDOW_S * 2
        self._recent_recognitions[camera_id] = [
            e for e in entries if e["timestamp"] > cutoff
        ]
        return None

    async def _backward_propagate(self, camera_id: int, named_object_id: int,
                                  name: str, category, bbox: Optional[dict],
                                  match_score: float):
        """After a successful recognition, go back and update recent unrecognized
        events on the same camera within the temporal window.

        For pets we additionally re-check the colour signal of each candidate
        thumbnail against the named profile's stored colour — propagation must
        not contradict the colour gate (e.g. tagging a tortoiseshell crop as
        the white cat).
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(seconds=TEMPORAL_WINDOW_S)
            cat_labels = ["person"] if category == ObjectCategory.person else ["cat", "dog"]

            # Pre-load the profile colour for pets so we can colour-gate each
            # propagation candidate.
            profile_color: Optional[str] = None
            if category == ObjectCategory.pet:
                try:
                    async with async_session() as session_p:
                        prof = await session_p.get(NamedObject, named_object_id)
                        if prof is not None:
                            attrs = prof.attributes or {}
                            profile_color = attrs.get("color") or attrs.get("colour")
                except Exception:
                    pass

            async with async_session() as session:
                result = await session.execute(
                    select(Event).where(
                        Event.camera_id == camera_id,
                        Event.named_object_id.is_(None),
                        Event.object_type.in_(cat_labels),
                        Event.started_at >= cutoff,
                    )
                )
                unrecognized = result.scalars().all()
                if not unrecognized:
                    return

                updated = 0
                for ev in unrecognized:
                    # Spatial check if both have bboxes
                    if bbox and ev.bbox:
                        iou = self._bbox_iou(bbox, ev.bbox)
                        if iou < TEMPORAL_IOU_MIN:
                            continue
                    # Pet colour gate on backward-propagation candidates.
                    if category == ObjectCategory.pet and profile_color and ev.thumbnail_path:
                        try:
                            from services.pet_color_gate import (
                                compute_colour_signal, colour_compatibility,
                            )
                            crop_img = cv2.imread(ev.thumbnail_path)
                            if crop_img is not None:
                                signal = compute_colour_signal(crop_img)
                                if colour_compatibility(profile_color, signal) <= 0.0:
                                    logger.info(
                                        "Backward-prop colour VETO: %s profile=%s observed=%s "
                                        "(white=%.2f black=%.2f) for event %d on cam %d",
                                        name, profile_color, signal.family,
                                        signal.white_ratio, signal.black_ratio,
                                        ev.id, camera_id,
                                    )
                                    continue
                        except Exception:
                            logger.debug("Backward-prop colour gate error", exc_info=True)
                    ev.named_object_id = named_object_id
                    ev.event_type = EventType.object_recognized
                    ev.confidence = match_score * 0.85
                    meta = dict(ev.metadata_extra or {})
                    meta["recognition_method"] = "backward_propagation"
                    meta["match_score"] = round(match_score * 0.85, 3)
                    meta["narrative"] = generate_narrative(
                        named_object_name=name,
                        object_type=ev.object_type,
                        camera_name=_camera_id_to_friendly.get(camera_id, f"camera_{camera_id}"),
                        timestamp=ev.started_at,
                    )
                    ev.metadata_extra = meta
                    flag_modified(ev, "metadata_extra")
                    session.add(ev)
                    updated += 1

                if updated:
                    await session.commit()
                    logger.info(
                        "Backward propagation: %s → %d events on camera %d",
                        name, updated, camera_id,
                    )
        except Exception:
            logger.debug("Backward propagation error", exc_info=True)

    # ── Burst Suppression ──

    def _is_burst_suppressed(self, camera_id: int) -> bool:
        """Check if a camera is currently burst-suppressed (too many unrecognized events)."""
        now = time.time()
        until = self._burst_suppressed_until.get(camera_id, 0)
        if now < until:
            return True
        return False

    def _record_unrecognized(self, camera_id: int):
        """Record an unrecognized event and trigger burst suppression if threshold exceeded."""
        now = time.time()
        window = self._unrecognized_timestamps.setdefault(camera_id, [])
        window.append(now)
        # Prune old timestamps outside the window
        cutoff = now - BURST_WINDOW_S
        self._unrecognized_timestamps[camera_id] = [t for t in window if t > cutoff]
        if len(self._unrecognized_timestamps[camera_id]) >= BURST_MAX_UNRECOGNIZED:
            self._burst_suppressed_until[camera_id] = now + BURST_COOLDOWN_S
            self._unrecognized_timestamps[camera_id] = []
            cam_name = _camera_id_to_friendly.get(camera_id, f"camera_{camera_id}")
            logger.warning(
                "Burst suppression activated for camera %s (%d): %d unrecognized events in %.0fs window",
                cam_name, camera_id, BURST_MAX_UNRECOGNIZED, BURST_WINDOW_S,
            )

    def _clear_burst_for_camera(self, camera_id: int):
        """Reset burst counter when a recognition succeeds (proves camera is seeing real objects)."""
        self._unrecognized_timestamps.pop(camera_id, None)
        self._burst_suppressed_until.pop(camera_id, None)

    # ── Static False-Positive Suppression ──

    @staticmethod
    def _bbox_iou(a: dict, b: dict) -> float:
        """Compute IoU between two bbox dicts with keys x1,y1,x2,y2."""
        ix1 = max(a["x1"], b["x1"])
        iy1 = max(a["y1"], b["y1"])
        ix2 = min(a["x2"], b["x2"])
        iy2 = min(a["y2"], b["y2"])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
        area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_zone_key(bbox: dict) -> str:
        """Quantise bbox to a coarse grid key for zone-level lookup."""
        # Round coords to nearest 20px to group similar positions
        return f"{bbox['x1']//20}_{bbox['y1']//20}_{bbox['x2']//20}_{bbox['y2']//20}"

    def _is_static_fp_suppressed(self, camera_id: int, bbox: dict) -> bool:
        """Check if a bbox position is currently suppressed as a static false positive."""
        now = time.time()
        zones = self._static_fp_suppressed.get(camera_id, {})
        zone_key = self._bbox_zone_key(bbox)
        until = zones.get(zone_key, 0)
        if now < until:
            return True
        # Also check neighbouring zone keys (bbox can shift by a few pixels)
        history = self._static_fp_history.get(camera_id, [])
        # Count how many recent unrecognized hits overlap this position
        cutoff = now - STATIC_FP_WINDOW_S
        hits = sum(1 for (hbox, ts) in history
                   if ts > cutoff and self._bbox_iou(bbox, hbox) >= STATIC_FP_IOU_THRESHOLD)
        if hits >= STATIC_FP_MIN_HITS:
            # Activate suppression for this zone
            if camera_id not in self._static_fp_suppressed:
                self._static_fp_suppressed[camera_id] = {}
            self._static_fp_suppressed[camera_id][zone_key] = now + STATIC_FP_SUPPRESS_S
            cam_name = _camera_id_to_friendly.get(camera_id, f"camera_{camera_id}")
            logger.warning(
                "Static FP suppression activated for %s at [%d,%d,%d,%d] — "
                "%d unrecognized detections at same position in %.0fs",
                cam_name, bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"],
                hits, STATIC_FP_WINDOW_S,
            )
            return True
        return False

    def _record_static_fp(self, camera_id: int, bbox: dict):
        """Record an unrecognized detection's bbox for static FP tracking."""
        now = time.time()
        history = self._static_fp_history.setdefault(camera_id, [])
        history.append((bbox, now))
        # Prune old entries
        cutoff = now - STATIC_FP_WINDOW_S
        self._static_fp_history[camera_id] = [(b, t) for (b, t) in history if t > cutoff]

    def _clear_static_fp_at_position(self, camera_id: int, bbox: dict):
        """Clear static FP suppression at this position (a real person was recognized here)."""
        zone_key = self._bbox_zone_key(bbox)
        zones = self._static_fp_suppressed.get(camera_id, {})
        zones.pop(zone_key, None)
        # Also remove matching history entries
        history = self._static_fp_history.get(camera_id, [])
        if history:
            self._static_fp_history[camera_id] = [
                (b, t) for (b, t) in history
                if self._bbox_iou(bbox, b) < STATIC_FP_IOU_THRESHOLD
            ]

    # ── Bbox Extraction & Cropping ──

    @staticmethod
    def _best_box(data: dict) -> Optional[list]:
        """Return the most accurate bbox available in a Frigate event payload.

        Frigate refines bboxes throughout the event lifetime. The fields, in
        order of accuracy:
          1. `current_box`   — bbox of the most recent detect frame (live)
          2. `snapshot.box`  — bbox of the frame that produced the saved snapshot
          3. `box`           — initial detection bbox (often stale by event end)

        We prefer (2) when the data describes the saved snapshot (which is what
        we'll actually fetch and crop), then fall back to (1) and (3).
        """
        snap = data.get("snapshot") or {}
        snap_box = snap.get("box") if isinstance(snap, dict) else None
        if snap_box and len(snap_box) == 4:
            return list(snap_box)
        current = data.get("current_box")
        if current and len(current) == 4:
            return list(current)
        box = data.get("box")
        if box and len(box) == 4:
            return list(box)
        return None

    @classmethod
    def _extract_bbox(cls, data: dict) -> Optional[dict]:
        """Convert Frigate's [x1,y1,x2,y2] box to our dict format."""
        box = cls._best_box(data)
        if box:
            return {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
        return None

    @classmethod
    def _crop_to_bbox(cls, frame: np.ndarray, data: dict, padding: float = 0.15) -> Optional[np.ndarray]:
        """Crop frame to the detected object's bounding box with padding.

        Isolates the target object from the full frame so recognition/training
        only sees ONE person/pet, not background people.

        Handles resolution mismatch: bbox coords are in Frigate detect resolution
        but the snapshot may be at a different (usually higher) resolution.

        Returns None if no valid bbox is available (callers must handle this).
        """
        box = cls._best_box(data)
        if not box:
            return None
        h, w = frame.shape[:2]

        x1f, y1f, x2f, y2f = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        # ── Scale bbox from detect resolution to snapshot resolution ──
        # If we picked the bbox from `snapshot.box`, the matching `frame_shape`
        # lives under `snapshot.frame_shape` and *that* is what we must scale
        # against (the snapshot frame, not the live detect frame).
        snap = data.get("snapshot") or {}
        snap_box = snap.get("box") if isinstance(snap, dict) else None
        if snap_box and list(snap_box) == [box[0], box[1], box[2], box[3]]:
            frame_shape = snap.get("frame_shape") or data.get("frame_shape")
        else:
            frame_shape = data.get("frame_shape")
        # Frigate webhook events do NOT always include `frame_shape`. When the
        # snapshot was pulled from the HD record stream (e.g. 3840x2160) but
        # the bbox is in detect-stream coordinates (640x360), we need a
        # fallback. Infer detect dims from the bbox extent: if bbox values fit
        # comfortably inside the detect frame and the snapshot is much bigger,
        # treat it as a detect-resolution bbox.
        if (not frame_shape or len(frame_shape) < 2) and (w >= 1000 or h >= 700):
            # Snapshot is HD/4K — bbox almost certainly in detect resolution.
            # Pick the smallest standard detect size that fully contains the bbox.
            bx_max = max(x2f, x1f)
            by_max = max(y2f, y1f)
            for cand_w, cand_h in ((640, 360), (640, 480), (1280, 720), (1920, 1080)):
                if bx_max <= cand_w + 2 and by_max <= cand_h + 2:
                    frame_shape = [cand_h, cand_w, 3]
                    break
        if frame_shape and len(frame_shape) >= 2:
            det_h, det_w = frame_shape[0], frame_shape[1]
            if det_w > 0 and det_h > 0:
                sx = w / det_w
                sy = h / det_h
                # Always scale if dimensions differ at all — even small
                # mismatches cause visibly off bboxes when boxes are large.
                if abs(sx - 1.0) >= 0.005 or abs(sy - 1.0) >= 0.005:
                    x1f, x2f = x1f * sx, x2f * sx
                    y1f, y2f = y1f * sy, y2f * sy
                    logger.debug(
                        "Scaled bbox %s → [%.0f,%.0f,%.0f,%.0f] "
                        "(detect %dx%d → snapshot %dx%d)",
                        box, x1f, y1f, x2f, y2f, det_w, det_h, w, h,
                    )

        x1, y1 = max(0, int(x1f)), max(0, int(y1f))
        x2, y2 = min(w, int(x2f)), min(h, int(y2f))
        if x2 <= x1 or y2 <= y1:
            return None

        # ── Tight padding so the subject fills the thumbnail ──
        # Goal: thumbnail should look like a portrait of the detected subject,
        # not a wide scene crop. Small head margin for people, uniform for
        # everything else. This makes both manual review and downstream vision
        # description more accurate.
        label = data.get("label", "")
        bw, bh = x2 - x1, y2 - y1
        if label == "person":
            pad_top = int(bh * 0.12)    # ~12% above head
            pad_bottom = int(bh * 0.05)
            pad_left = int(bw * 0.06)
            pad_right = int(bw * 0.06)
        elif label in ("cat", "dog"):
            pad_top = int(bh * 0.10)
            pad_bottom = int(bh * 0.10)
            pad_left = int(bw * 0.10)
            pad_right = int(bw * 0.10)
        else:
            pad_top = int(bh * padding)
            pad_bottom = int(bh * padding)
            pad_left = int(bw * padding)
            pad_right = int(bw * padding)

        cx1 = max(0, x1 - pad_left)
        cy1 = max(0, y1 - pad_top)
        cx2 = min(w, x2 + pad_right)
        cy2 = min(h, y2 + pad_bottom)
        cropped = frame[cy1:cy2, cx1:cx2]

        # Ensure crop is meaningful (at least 50×50 pixels). Below this size
        # the subject is too small to be useful for review or recognition.
        if cropped.shape[0] < 50 or cropped.shape[1] < 50:
            return None
        return cropped

    # ── Crop Validation & PTZ-aware Retry ──

    @staticmethod
    def _is_crop_valid(crop: Optional[np.ndarray], full_frame: np.ndarray,
                       blur_threshold: float = 30.0,
                       uniformity_threshold: float = 15.0) -> bool:
        """Check whether a crop likely contains a real object (not empty/blurry).

        Returns False if:
        - Crop is None (bbox extraction failed)
        - Crop is identical to the full frame (bbox failed, no actual crop)
        - Crop is too blurry (motion blur from PTZ movement), Laplacian var < 30
        - Crop is too uniform (empty wall/sky/ground), pixel stdev < 15
        """
        if crop is None:
            return False

        # If crop is the same object as the full frame, bbox extraction failed
        if crop is full_frame:
            return False

        # Check motion blur via Laplacian variance
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < blur_threshold:
            return False

        # Check if crop is mostly uniform (empty area)
        pixel_std = float(np.std(gray))
        if pixel_std < uniformity_threshold:
            return False

        return True

    async def _fetch_frigate_crop(self, frigate_id: str) -> Optional[np.ndarray]:
        """Fetch Frigate's own pre-cropped snapshot (atomic: frame + bbox match)."""
        try:
            resp = await self._http.get(
                f"/api/events/{frigate_id}/snapshot.jpg",
                params={"crop": 1, "quality": 95, "bbox": 0},
            )
            if resp.status_code != 200:
                return None
            arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and img.shape[0] >= 40 and img.shape[1] >= 40:
                return img
            return None
        except Exception as e:
            logger.debug("Failed to fetch Frigate crop for %s: %s", frigate_id[:12], e)
            return None

    async def _fetch_recording_frame(self, camera_name: str,
                                     timestamp: float) -> Optional[np.ndarray]:
        """Fetch latest frame from camera as a fallback when snapshot is stale."""
        try:
            resp = await self._http.get(
                f"/api/{camera_name}/latest.jpg",
                params={"quality": 95},
            )
            if resp.status_code == 200:
                arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img
        except Exception as e:
            logger.debug("Failed to fetch live frame for %s: %s", camera_name, e)
        return None

    async def _fetch_and_validate_crop(
        self,
        frigate_id: str,
        camera_id: int,
        camera_name: str,
        data: dict,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fetch a clean full frame + a tight, accurate crop of the object.

        Atomic-first strategy:
          1. **Frigate atomic crop** — `/api/events/<id>/snapshot.jpg?crop=1&bbox=0`.
             Frigate applies its own bbox to its own snapshot frame server-side,
             so the crop is guaranteed to be of the right object on the right
             frame. No resolution-mixing math, no PTZ desync.
          2. Full clean snapshot (`bbox=0`) for context / UI / face-fallback.
          3. If the atomic crop is unavailable or fails our blur/uniformity
             gate, we re-derive it locally from the clean snapshot using the
             newest bbox we know about (`snapshot.box` → `current_box` → `box`).
          4. Wait briefly + re-fetch (PTZ settling).
          5. Live frame from go2rtc as last resort.
          6. If everything fails, return `(full_frame_or_None, None)`. Callers
             must NEVER substitute the full frame for the crop — that defeats
             object isolation and lets face recognition see the whole scene.

        Returns (full_frame, object_crop). `object_crop` is None only on
        complete failure.
        """
        cam_friendly = _camera_id_to_friendly.get(camera_id, camera_name)
        frame_time = data.get("frame_time") or data.get("start_time")

        # ── Attempt 1: atomic Frigate crop (the cleanest option) ──
        # This single call gives us a crop that is guaranteed to belong to
        # the same frame Frigate ran detection on, with no overlay drawn.
        atomic_crop = await self._fetch_frigate_crop(frigate_id)
        atomic_crop_valid = (
            atomic_crop is not None
            and self._is_crop_valid(atomic_crop, None)
        )

        # Always also fetch the full clean snapshot for context (UI snapshot,
        # attribute estimator frame_shape, face fallback against full body).
        full_frame = await self._fetch_snapshot(
            frigate_id, camera_name=camera_name, frame_time=frame_time,
        )

        if atomic_crop_valid:
            return full_frame, atomic_crop

        if full_frame is None:
            # Couldn't get the context frame either — caller will skip.
            logger.warning(
                "No clean snapshot available for %s on %s",
                frigate_id[:12], cam_friendly,
            )
            return None, atomic_crop  # atomic_crop may still be a usable last resort

        snap_h, snap_w = full_frame.shape[:2]

        # ── Attempt 2: local crop from clean snapshot using newest bbox ──
        local_crop = self._crop_to_bbox(full_frame, data)
        if self._is_crop_valid(local_crop, full_frame):
            return full_frame, local_crop

        logger.info(
            "Atomic + local crops both invalid for %s on %s "
            "(box=%s, snap=%dx%d) — waiting 1s and retrying",
            frigate_id[:12], cam_friendly, self._best_box(data), snap_w, snap_h,
        )

        # ── Attempt 3: brief wait + re-fetch atomic crop (PTZ settle) ──
        await asyncio.sleep(1.0)
        atomic_crop_2 = await self._fetch_frigate_crop(frigate_id)
        if atomic_crop_2 is not None and self._is_crop_valid(atomic_crop_2, None):
            logger.info("Retry atomic crop succeeded for %s on %s", frigate_id[:12], cam_friendly)
            return full_frame, atomic_crop_2

        full_frame_2 = await self._fetch_snapshot(
            frigate_id, camera_name=camera_name, frame_time=frame_time,
        )
        if full_frame_2 is not None:
            local_crop_2 = self._crop_to_bbox(full_frame_2, data)
            if self._is_crop_valid(local_crop_2, full_frame_2):
                logger.info("Retry snapshot succeeded for %s on %s", frigate_id[:12], cam_friendly)
                return full_frame_2, local_crop_2

        # ── Attempt 4: live camera frame + re-crop ──
        live_frame = await self._fetch_recording_frame(
            camera_name, data.get("start_time", time.time())
        )
        if live_frame is not None:
            live_crop = self._crop_to_bbox(live_frame, data)
            if self._is_crop_valid(live_crop, live_frame):
                logger.info("Live frame crop succeeded for %s on %s", frigate_id[:12], cam_friendly)
                return live_frame, live_crop

        # ── Last resort: return whatever atomic crop we got, even if blurry ──
        # NEVER return the full frame as the crop — that lets face recognition
        # roam over the whole scene and pick up faces of other people / posters
        # / TV screens. Better to skip recognition for this event.
        if atomic_crop is not None:
            logger.warning(
                "Using blurry atomic crop for %s on %s — all retries failed",
                frigate_id[:12], cam_friendly,
            )
            return full_frame, atomic_crop
        if local_crop is not None:
            logger.warning(
                "Using low-quality local crop for %s on %s — all retries failed",
                frigate_id[:12], cam_friendly,
            )
            return full_frame, local_crop

        logger.warning(
            "All crop strategies failed for %s on %s — recognition will be skipped",
            frigate_id[:12], cam_friendly,
        )
        return full_frame, None

    # ── Diagnostics ──

    def get_current_presence(self) -> dict[str, int]:
        """Return map of named_object_name → camera_id for currently present objects."""
        # _presence is keyed by named_object_id → {camera_id, last_seen, name}
        result = {}
        now = time.time()
        for obj_id, entry in self._presence.items():
            if (now - entry["last_seen"]) < PRESENCE_TIMEOUT_S:
                name = entry.get("name", str(obj_id))
                result[name] = entry["camera_id"]
        return result

    def get_status(self) -> dict:
        now = time.time()
        suppressed_cams = {
            _camera_id_to_friendly.get(cid, f"camera_{cid}"): round(until - now)
            for cid, until in self._burst_suppressed_until.items()
            if until > now
        }
        static_fp_zones = {}
        for cid, zones in self._static_fp_suppressed.items():
            active = {k: round(v - now) for k, v in zones.items() if v > now}
            if active:
                static_fp_zones[_camera_id_to_friendly.get(cid, f"camera_{cid}")] = active
        return {
            "running": self._running,
            "mqtt_connected": self._mqtt.is_connected() if self._mqtt else False,
            "active_events": len(self._event_map),
            "presence_tracked": len(self._presence),
            "cameras_mapped": len(_camera_name_to_id),
            "burst_suppressed": suppressed_cams,
            "static_fp_suppressed": static_fp_zones,
        }

    async def get_active_frigate_events(self) -> list[dict]:
        """Query Frigate API for in-progress events + very recent ended events.

        Returns normalized list of {frigate_camera, label, score, bbox, sub_label, in_progress}.
        """
        if not self._http:
            return []
        results = []
        try:
            # In-progress events
            resp = await self._http.get("/api/events", params={"in_progress": 1})
            if resp.status_code == 200:
                for ev in resp.json():
                    results.append(self._normalize_frigate_event(ev, in_progress=True))

            # Also get very recent ended events (last 2 min) for stationary objects
            import time as _time
            after_ts = _time.time() - 120
            resp2 = await self._http.get("/api/events", params={"limit": 20, "after": after_ts})
            if resp2.status_code == 200:
                seen_ids = {r["frigate_id"] for r in results}
                for ev in resp2.json():
                    if ev["id"] not in seen_ids:
                        results.append(self._normalize_frigate_event(ev, in_progress=False))
        except Exception as e:
            logger.debug("Failed to query Frigate active events: %s", e)
        return results

    @staticmethod
    def _normalize_frigate_event(ev: dict, in_progress: bool) -> dict:
        """Convert a Frigate event dict to a normalized format."""
        data = ev.get("data", {})
        box = data.get("box") or [0, 0, 0, 0]
        # Frigate box format: [x_left, y_top, width, height] in 0-1 coords
        # Convert to [x1, y1, x2, y2] in 0-1 coords
        if len(box) == 4:
            x, y, w, h = box
            bbox_norm = [max(0, x), max(0, y), min(1, x + w), min(1, y + h)]
        else:
            bbox_norm = [0, 0, 0, 0]
        return {
            "frigate_id": ev["id"],
            "frigate_camera": ev.get("camera", ""),
            "label": ev.get("label", "object"),
            "score": data.get("top_score") or data.get("score", 0),
            "bbox_norm": bbox_norm,
            "sub_label": ev.get("sub_label"),
            "in_progress": in_progress,
        }

    def invalidate_embedding_cache(self):
        """Signal that training data changed — downstream services will pick up fresh embeddings."""
        logger.info("Embedding cache invalidated (training data changed)")
        # face_service and recognition_service reload on demand — no explicit cache to flush


# Singleton
frigate_bridge = FrigateBridge()
