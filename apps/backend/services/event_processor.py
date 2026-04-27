"""BanusNas — Event Processor: orchestrates tracking → recognition → storage → notify."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.database import async_session
from models.schemas import Camera, Event, EventType, NamedObject, ObjectCategory
from services.object_detector import Detection, object_detector
from services.recognition_service import recognition_service
from services.face_service import face_service
from services.object_tracker import object_tracker
from services.attribute_estimator import (
    estimate_person_attributes,
    merge_stable_attributes,
    compute_attribute_multiplier,
    get_display_attributes,
)
from services.recognition_agent import recognition_agent
from services.narrative_generator import generate_narrative

logger = logging.getLogger(__name__)

# ── Configurable at runtime via performance settings ──
_JPEG_QUALITY = 80


def _hot_snapshots_path() -> str:
    """Return hot storage snapshots path if available, else cold."""
    if settings.hot_storage_path:
        return str(Path(settings.hot_storage_path) / "snapshots")
    return settings.snapshots_path


class EventBus:
    """Simple async pub/sub for real-time event distribution to WebSocket clients."""

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.remove(q)

    async def publish(self, event: dict):
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # Drop if client is slow


# Global event bus
event_bus = EventBus()


class EventProcessor:
    """Orchestrates the tracking → recognition → storage → notification pipeline."""

    def __init__(self):
        self._notification_engine = None  # Set after import to avoid circular deps
        self._detection_semaphore = asyncio.Semaphore(2)  # Limit concurrent recognition pipelines
        # Maps (camera_id, track_id) → event_id for the single-event lifecycle
        self._track_event_map: dict[tuple[int, int], int] = {}
        # In-memory caches for known objects (avoid DB roundtrip per detection)
        self._person_face_cache: list[tuple[int, str, list[float]]] = []  # (id, name, embedding)
        self._person_body_cache: list[tuple[int, str, list[float]]] = []  # (id, name, body_embedding)
        self._pet_cache: dict[str, list[tuple[int, str, list[float]]]] = {}  # category → [(id, name, embedding)]
        self._person_attributes_cache: dict[int, dict] = {}  # named_object_id → attributes dict
        self._false_positive_cache: list[list[float]] = []  # body embeddings of false positive "persons"
        self._cache_ts: float = 0.0
        self._CACHE_TTL = 30.0  # Refresh every 30s

        # ── Event consolidation: avoid duplicate events for stationary objects ──
        # Key: (camera_id, object_type, named_object_id_or_"unknown")
        # Value: {event_id, ended_at, bbox, last_extended}
        self._recent_events: dict[tuple, dict] = {}

        # ── Cross-camera presence: track where named objects currently are ──
        # Key: named_object_id → {camera_id, since, last_seen}
        self._presence: dict[int, dict] = {}

        # ── Static object suppression ──
        # Detects stationary objects (statues, garden gnomes, etc.) that YOLO
        # repeatedly classifies as "person" but never match any face/body.
        # Key: (camera_id, quantized_bbox) → {count, first_seen, last_seen}
        self._static_ghosts: dict[tuple, dict] = {}
        self._STATIC_GHOST_IOU = 0.60      # IoU threshold to consider "same spot"
        self._STATIC_GHOST_TRIGGER = 3     # Detections before suppression kicks in
        self._STATIC_GHOST_WINDOW = 3600   # 1 hour — detections within this window count
        self._STATIC_GHOST_SUPPRESS = 7200 # 2 hours — suppress for this long after trigger

    def set_notification_engine(self, engine):
        self._notification_engine = engine

    def _collect_event_annotations(
        self,
        camera_id: int,
        det: Detection,
        named_object_name: Optional[str],
    ) -> list[dict]:
        annotations: list[dict] = []
        seen_keys: set[tuple[str, int, int, int, int]] = set()

        def add_annotation(name: Optional[str], class_name: str, bbox: tuple[int, int, int, int], confidence: float, primary: bool) -> None:
            if not name:
                return
            key = (name, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            if key in seen_keys:
                return
            seen_keys.add(key)
            annotations.append({
                "name": name,
                "class_name": class_name,
                "bbox": {
                    "x1": int(bbox[0]),
                    "y1": int(bbox[1]),
                    "x2": int(bbox[2]),
                    "y2": int(bbox[3]),
                },
                "confidence": round(float(confidence), 3),
                "primary": primary,
            })

        add_annotation(named_object_name, det.class_name, det.bbox, det.confidence, True)

        active_tracks = object_tracker.get_active_tracks(camera_id)
        for track in active_tracks.values():
            track_name = track.get("named_object_name")
            bbox = track.get("bbox")
            if not track_name or not bbox or len(bbox) != 4:
                continue
            add_annotation(
                track_name,
                str(track.get("class_name") or det.class_name),
                tuple(int(v) for v in bbox),
                float(track.get("confidence") or 0.0),
                False,
            )

        return annotations

    async def _refresh_embedding_cache(self):
        """Reload known object embeddings from DB into memory. Called lazily."""
        import time
        now = time.monotonic()
        if now - self._cache_ts < self._CACHE_TTL:
            return
        self._cache_ts = now
        try:
            async with async_session() as session:
                # Person face embeddings
                result = await session.execute(
                    select(NamedObject).where(
                        NamedObject.category == ObjectCategory.person,
                        NamedObject.embedding.isnot(None),
                    )
                )
                self._person_face_cache = [
                    (p.id, p.name, p.embedding) for p in result.scalars().all()
                ]
                # Person body embeddings
                result = await session.execute(
                    select(NamedObject).where(
                        NamedObject.category == ObjectCategory.person,
                        NamedObject.body_embedding.isnot(None),
                    )
                )
                self._person_body_cache = [
                    (p.id, p.name, p.body_embedding) for p in result.scalars().all()
                ]
                # Person soft biometric attributes
                result = await session.execute(
                    select(NamedObject).where(
                        NamedObject.category == ObjectCategory.person,
                    )
                )
                self._person_attributes_cache = {
                    p.id: p.attributes for p in result.scalars().all() if p.attributes
                }
                # Pet/vehicle/other embeddings by category
                for cat in [ObjectCategory.pet, ObjectCategory.vehicle, ObjectCategory.other]:
                    result = await session.execute(
                        select(NamedObject).where(
                            NamedObject.category == cat,
                            NamedObject.embedding.isnot(None),
                        )
                    )
                    self._pet_cache[cat.value] = [
                        (p.id, p.name, p.embedding) for p in result.scalars().all()
                    ]
                # False positive embeddings (not-a-person negative examples)
                from models.schemas import SystemSettings
                fp_result = await session.execute(
                    select(SystemSettings).where(SystemSettings.key == "false_positive_embeddings")
                )
                fp_setting = fp_result.scalar_one_or_none()
                if fp_setting and fp_setting.value.get("embeddings"):
                    self._false_positive_cache = fp_setting.value["embeddings"]
                else:
                    self._false_positive_cache = []
            logger.debug(
                "Embedding cache refreshed: %d faces, %d bodies, %d pets, %d attribute profiles, %d false positives",
                len(self._person_face_cache), len(self._person_body_cache),
                sum(len(v) for v in self._pet_cache.values()),
                len(self._person_attributes_cache),
                len(self._false_positive_cache),
            )
        except Exception as e:
            logger.warning("Embedding cache refresh failed: %s", e)

    def invalidate_embedding_cache(self):
        """Force cache refresh on next detection (called after training)."""
        self._cache_ts = 0.0

    # ── Event consolidation settings ──
    # How long after an event ends before we consider a new detection a fresh event
    CONSOLIDATION_GAP_NAMED = 1800      # 30 min for recognised objects
    CONSOLIDATION_GAP_UNKNOWN = 120     # 2 min for unknowns (same spot on same camera)
    CONSOLIDATION_IOU_THRESHOLD = 0.20  # Bbox overlap needed to consolidate
    # Persistent presence: after this many consolidations, use extended gap
    PERSISTENT_PRESENCE_COUNT = 3
    CONSOLIDATION_GAP_PERSISTENT = 7200  # 2 hr for persistent presence (sleeping pet, parked car)

    # ── Cross-camera presence settings ──
    PRESENCE_TIMEOUT = 1800  # 30 min — after this, presence is stale

    @staticmethod
    def _bbox_iou(b1: dict, b2: dict) -> float:
        """Compute IoU between two bbox dicts (keys: x1,y1,x2,y2 or 0,1,2,3)."""
        def _unpack(b):
            return (
                float(b.get("x1", b.get("0", 0))),
                float(b.get("y1", b.get("1", 0))),
                float(b.get("x2", b.get("2", 0))),
                float(b.get("y2", b.get("3", 0))),
            )
        ax1, ay1, ax2, ay2 = _unpack(b1)
        bx1, by1, bx2, by2 = _unpack(b2)
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _try_consolidate(
        self, camera_id: int, object_type: str,
        named_object_id: Optional[int], bbox: dict,
    ) -> Optional[int]:
        """Check if a new detection should extend an existing event.

        Returns the event_id to extend, or None if a new event is needed.
        """
        import time
        now = time.time()
        oid_key = named_object_id if named_object_id else "unknown"
        key = (camera_id, object_type, oid_key)

        cached = self._recent_events.get(key)
        if not cached:
            return None

        # Time since last detection on this event
        elapsed = now - cached["last_extended"]

        # Named objects get a longer gap (the cat hasn't moved from the sofa)
        if named_object_id:
            # Persistent presence: if consolidated many times, use extended gap
            consolidation_count = cached.get("consolidation_count", 1)
            if consolidation_count >= self.PERSISTENT_PRESENCE_COUNT:
                gap_limit = self.CONSOLIDATION_GAP_PERSISTENT
            else:
                gap_limit = self.CONSOLIDATION_GAP_NAMED
        else:
            gap_limit = self.CONSOLIDATION_GAP_UNKNOWN

        if elapsed > gap_limit:
            # Too long — this is a fresh visit
            del self._recent_events[key]
            return None

        # Check spatial overlap — same area of the camera frame?
        stored_bbox = cached.get("bbox")
        if stored_bbox and bbox:
            iou = self._bbox_iou(stored_bbox, bbox)
            if iou < self.CONSOLIDATION_IOU_THRESHOLD:
                return None  # Different position — new event

        return cached["event_id"]

    def _record_event(
        self, camera_id: int, object_type: str,
        named_object_id: Optional[int], event_id: int, bbox: dict,
    ):
        """Record a new/extended event in the consolidation cache."""
        import time
        oid_key = named_object_id if named_object_id else "unknown"
        key = (camera_id, object_type, oid_key)
        existing = self._recent_events.get(key)
        count = (existing.get("consolidation_count", 1) + 1) if existing and existing["event_id"] == event_id else 1
        self._recent_events[key] = {
            "event_id": event_id,
            "bbox": bbox,
            "last_extended": time.time(),
            "consolidation_count": count,
        }

    def update_presence(self, named_object_id: int, camera_id: int):
        """Update the cross-camera presence record for a named object."""
        import time
        now = time.time()
        existing = self._presence.get(named_object_id)
        if existing and existing["camera_id"] == camera_id:
            existing["last_seen"] = now
        else:
            self._presence[named_object_id] = {
                "camera_id": camera_id,
                "since": now,
                "last_seen": now,
            }

    def get_presence(self, named_object_id: int) -> Optional[dict]:
        """Get where a named object was last seen. Returns None if stale."""
        import time
        entry = self._presence.get(named_object_id)
        if not entry:
            return None
        if time.time() - entry["last_seen"] > self.PRESENCE_TIMEOUT:
            return None
        return entry

    def is_presence_conflict(self, named_object_id: int, camera_id: int) -> bool:
        """Check if a named object is believed to be on a DIFFERENT camera right now.

        Returns True if the object was seen on another camera recently enough
        that this ID match is suspicious.
        """
        import time
        entry = self._presence.get(named_object_id)
        if not entry:
            return False
        if entry["camera_id"] == camera_id:
            return False
        # How recently were they seen on the other camera?
        age = time.time() - entry["last_seen"]
        # Within 60s on a different camera — very likely a false match
        if age < 60:
            return True
        # 1-5 min — possible but suspicious (person could have walked)
        # For non-persons (cats, cars) give a longer window
        return False

    # ─── Static object (ghost) detection ───

    def _static_ghost_key(self, camera_id: int, det: Detection) -> Optional[tuple]:
        """Find an existing ghost entry whose bbox overlaps this detection, or return a new key."""
        import time
        now = time.time()
        bbox_dict = {"x1": det.bbox[0], "y1": det.bbox[1], "x2": det.bbox[2], "y2": det.bbox[3]}
        # Check existing ghosts for this camera
        for key, info in list(self._static_ghosts.items()):
            if key[0] != camera_id:
                continue
            # Expire old entries
            if now - info["last_seen"] > self._STATIC_GHOST_SUPPRESS:
                del self._static_ghosts[key]
                continue
            stored_bbox = info["bbox"]
            iou = self._bbox_iou(bbox_dict, stored_bbox)
            if iou >= self._STATIC_GHOST_IOU:
                return key
        # No match — create new key from quantized bbox center
        cx = (det.bbox[0] + det.bbox[2]) // 2
        cy = (det.bbox[1] + det.bbox[3]) // 2
        return (camera_id, cx // 50, cy // 50, det.class_name)

    def _is_static_ghost(self, camera_id: int, det: Detection) -> bool:
        """Check if this detection is at a known static-object position.

        Returns True (suppress) if this camera+position has produced
        STATIC_GHOST_TRIGGER or more unnamed detections within the window.
        """
        import time
        key = self._static_ghost_key(camera_id, det)
        if key is None:
            return False
        info = self._static_ghosts.get(key)
        if info is None:
            return False
        now = time.time()
        # Is the ghost active (enough detections within the window)?
        if info["count"] >= self._STATIC_GHOST_TRIGGER:
            if now - info["first_seen"] < self._STATIC_GHOST_SUPPRESS:
                logger.info(
                    "Static ghost suppressed: camera=%d %s at (%d,%d)-(%d,%d) "
                    "(seen %d times, first %.0fs ago)",
                    camera_id, det.class_name,
                    det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
                    info["count"], now - info["first_seen"],
                )
                return True
        return False

    def _record_static_ghost(self, camera_id: int, det: Detection):
        """Record an unnamed detection for static ghost tracking.

        Only called when the detection was NOT matched to any named object.
        """
        import time
        now = time.time()
        key = self._static_ghost_key(camera_id, det)
        if key is None:
            return
        bbox_dict = {"x1": det.bbox[0], "y1": det.bbox[1], "x2": det.bbox[2], "y2": det.bbox[3]}
        info = self._static_ghosts.get(key)
        if info is None:
            self._static_ghosts[key] = {
                "bbox": bbox_dict,
                "count": 1,
                "first_seen": now,
                "last_seen": now,
            }
        else:
            # Reset if first detection is too old
            if now - info["first_seen"] > self._STATIC_GHOST_WINDOW:
                info["count"] = 1
                info["first_seen"] = now
            else:
                info["count"] += 1
            info["last_seen"] = now
            info["bbox"] = bbox_dict  # Update to latest bbox

    def _clear_static_ghost(self, camera_id: int, det: Detection):
        """Clear a ghost entry when a named match is found (not static after all)."""
        key = self._static_ghost_key(camera_id, det)
        if key and key in self._static_ghosts:
            del self._static_ghosts[key]

    # ─── Continuous tracking callbacks ───

    async def on_object_appear(self, camera_id: int, camera_name: str, frame: np.ndarray, det: Detection, track_id: int = 0):
        """Called by ObjectTracker when a confirmed object first appears.

        Creates a single Event record that will be updated throughout the track's life.
        """
        try:
            async with async_session() as session:
                camera = (await session.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
                if camera:
                    camera_name = camera.name
            event_id = await self._process_detection(camera_id, camera_name, frame, det)
            if event_id and track_id:
                self._track_event_map[(camera_id, track_id)] = event_id
                object_tracker.set_track_event_id(camera_id, track_id, event_id)
        except Exception as e:
            logger.error("Object appear processing failed for camera %s: %s", camera_name, e)

    async def on_object_depart(self, camera_id: int, camera_name: str, track):
        """Called by ObjectTracker when a tracked object disappears.

        Updates the original event with ended_at and builds a GIF timelapse
        from collected crops.
        """
        try:
            async with async_session() as session:
                camera = (await session.execute(select(Camera).where(Camera.id == camera_id))).scalar_one_or_none()
                if camera:
                    camera_name = camera.name

            duration = round(track.last_seen - track.first_seen, 1)
            event_id = self._track_event_map.pop((camera_id, track.track_id), None) or getattr(track, 'event_id', None)

            # Update the original event with ended_at and GIF
            if event_id:
                gif_path = None
                if len(track.gif_crops) >= 2:
                    gif_path = await self._build_gif(camera_id, track.gif_crops, track.class_name)

                async with async_session() as session:
                    event = await session.get(Event, event_id)
                    if event:
                        event.ended_at = datetime.now(timezone.utc)
                        if gif_path:
                            meta = event.metadata_extra or {}
                            meta["gif_path"] = gif_path
                            event.metadata_extra = meta
                        await session.commit()

            logger.info(
                "Object departed: camera=%s type=%s name=%s duration=%.1fs event=%s gif=%s",
                camera_name, track.class_name, track.named_object_name or "unknown",
                duration, event_id, bool(track.gif_crops) and len(track.gif_crops) >= 2,
            )
            await event_bus.publish({
                "type": "object_departed",
                "camera_id": camera_id,
                "camera_name": camera_name,
                "object_type": track.class_name,
                "named_object_name": track.named_object_name,
                "duration": duration,
                "event_id": event_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.error("Object depart processing failed for camera %s: %s", camera_name, e)

    async def on_tracking_snapshot(self, camera_id: int, camera_name: str, frame: np.ndarray, det: Detection, track_id: int = 0):
        """Called periodically for unnamed tracked objects — retry recognition.

        No new events are created. If recognition succeeds, the original event
        is updated with the identity (named_object_id, name, thumbnail).
        """
        try:
            event_id = self._track_event_map.get((camera_id, track_id))
            if not event_id:
                return

            # Only retry recognition for unnamed objects
            async with self._detection_semaphore:
                result = await self._try_recognize(camera_id, camera_name, frame, det)

            if result and result[0] is not None:
                named_object_id, named_object_name, embedding = result

                # Update existing event with the identity
                thumbnail_path = await self._save_thumbnail(camera_id, det, frame)
                async with async_session() as session:
                    event = await session.get(Event, event_id)
                    if event and not event.named_object_id:
                        event.event_type = EventType.object_recognized
                        event.named_object_id = named_object_id
                        event.thumbnail_path = thumbnail_path
                        if embedding:
                            meta = event.metadata_extra or {}
                            meta["embedding"] = embedding if isinstance(embedding, list) else embedding.tolist()
                            event.metadata_extra = meta
                        await session.commit()

                # Write name back to live tracker
                object_tracker.set_track_name(camera_id, det.class_name, det.bbox, named_object_name)

                logger.info(
                    "Late recognition: camera=%s type=%s name=%s (event %d)",
                    camera_name, det.class_name, named_object_name, event_id,
                )

                # Send notification for late recognition
                if self._notification_engine and not object_tracker.is_named_object_recently_departed(camera_id, named_object_name):
                    await self._notification_engine.evaluate_and_notify(
                        camera_id=camera_id, camera_name=camera_name,
                        object_type=det.class_name, named_object_id=named_object_id,
                        named_object_name=named_object_name, event_id=event_id,
                        snapshot_path=None,
                    )
        except Exception as e:
            logger.error("Tracking snapshot failed for camera %s: %s", camera_name, e)

    async def _enhanced_detect(self, frame: np.ndarray, motion_regions: list, enhanced_classes: list[str], detection_settings: dict, global_confidence: float) -> list[Detection]:
        """Run detection on cropped motion regions for classes with 'enhanced' enabled.

        This helps catch small objects (e.g. cats, distant people) that may be hard
        to detect at full-frame resolution.
        """
        extra = []
        h, w = frame.shape[:2]
        min_conf = min(
            detection_settings.get(cls, {}).get("confidence", global_confidence)
            for cls in enhanced_classes
        )

        for region in motion_regions[:4]:  # Limit to 4 regions to avoid excessive compute
            rx, ry, rw, rh = region
            # Pad region by 50% for context
            pad = max(rw, rh) // 2
            x1 = max(0, rx - pad)
            y1 = max(0, ry - pad)
            x2 = min(w, rx + rw + pad)
            y2 = min(h, ry + rh + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 60 or crop.shape[1] < 60:
                continue

            dets = await object_detector.detect(crop, confidence_threshold=min_conf, target_classes=enhanced_classes)
            for d in dets:
                # Remap bbox from crop coordinates back to full frame
                bx1, by1, bx2, by2 = d.bbox
                abs_bbox = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
                # Re-crop from full frame at mapped coordinates
                fx1, fy1, fx2, fy2 = abs_bbox
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(w, fx2), min(h, fy2)
                if fx2 > fx1 and fy2 > fy1:
                    full_crop = frame[fy1:fy2, fx1:fx2].copy()
                    extra.append(Detection(d.class_name, d.confidence, (fx1, fy1, fx2, fy2), full_crop))

        return extra

    @staticmethod
    def _nms_detections(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
        """Apply NMS across a list of Detection objects to remove duplicates."""
        import cv2 as cv2_nms
        boxes = np.array([d.bbox for d in detections], dtype=np.float32)
        scores = np.array([d.confidence for d in detections], dtype=np.float32)
        indices = cv2_nms.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return detections

    async def _is_false_positive(self, crop: np.ndarray) -> bool:
        """Check if a person crop matches any known false positive (not-a-person) embeddings."""
        try:
            body_emb = recognition_service.compute_reid_embedding(crop)
            if not body_emb:
                return False
            emb = np.array(body_emb, dtype=np.float32)
            for fp_emb in self._false_positive_cache:
                stored = np.array(fp_emb, dtype=np.float32)
                sim = float(np.dot(emb, stored) / max(np.linalg.norm(emb) * np.linalg.norm(stored), 1e-8))
                if sim >= 0.70:
                    return True
        except Exception as e:
            logger.debug("False positive check failed: %s", e)
        return False

    async def _process_detection(
        self, camera_id: int, camera_name: str, frame: np.ndarray, det: Detection
    ) -> Optional[int]:
        """Process a single detected object: recognize, store, notify. Returns event_id."""
        async with self._detection_semaphore:
            return await self._process_detection_inner(camera_id, camera_name, frame, det)

    async def _process_detection_inner(
        self, camera_id: int, camera_name: str, frame: np.ndarray, det: Detection
    ):
        """Inner detection processing (rate-limited by semaphore)."""

        # Refresh embedding cache (lazy, respects TTL)
        await self._refresh_embedding_cache()

        # ── Agent quality gate: reject low-quality detections early ──
        from services.recognition_agent import MIN_CONFIDENCE_FOR_RECOGNITION
        if det.confidence < MIN_CONFIDENCE_FOR_RECOGNITION:
            logger.debug("Suppressed low-confidence %s (%.2f) on %s", det.class_name, det.confidence, camera_name)
            return None

        crop_quality = recognition_agent.assess_crop_quality(det.crop, det.class_name)
        if not crop_quality.is_valid:
            logger.info(
                "Agent rejected %s crop on %s: %s",
                det.class_name, camera_name, crop_quality.reject_reason,
            )
            return None

        # ── Static object suppression ──
        # Suppress detections at positions where previous detections were
        # always unrecognised — likely a statue, mannequin, or garden ornament.
        if self._is_static_ghost(camera_id, det):
            return None

        # Map COCO class names to our ObjectCategory for embedding lookup
        EMBEDDING_CATEGORY_MAP = {
            "cat": ObjectCategory.pet,
            "dog": ObjectCategory.pet,
            "car": ObjectCategory.vehicle,
            "truck": ObjectCategory.vehicle,
            "bus": ObjectCategory.vehicle,
            "motorcycle": ObjectCategory.vehicle,
            "bicycle": ObjectCategory.vehicle,
            "boat": ObjectCategory.vehicle,
            "train": ObjectCategory.vehicle,
            "airplane": ObjectCategory.vehicle,
        }

        # ─── Person detection: face-based recognition ───
        if det.class_name == "person":
            # Check against false positive (not-a-person) embeddings
            if self._false_positive_cache and await self._is_false_positive(det.crop):
                logger.info("Suppressed false positive person detection on %s", camera_name)
                return None

            # Detect faces in the person crop
            faces = await face_service.detect_faces_async(det.crop)

            # Retry with upscaled crop for small/distant person detections
            if not faces:
                h, w = det.crop.shape[:2]
                if h < 200 or w < 120:
                    scale = min(2.5, max(200 / h, 120 / w))
                    upscaled = cv2.resize(det.crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    up_faces = await face_service.detect_faces_async(upscaled)
                    if up_faces:
                        for f in up_faces:
                            bx1, by1, bx2, by2 = f.bbox
                            f.bbox = (int(bx1 / scale), int(by1 / scale), int(bx2 / scale), int(by2 / scale))
                        faces = up_faces
                        logger.debug("Face found after upscale (%.1fx) on %s", scale, camera_name)

            if len(faces) >= 2:
                # Multi-person box — split into one event per face
                logger.info(
                    "Multi-face box (%d faces) on %s, splitting",
                    len(faces), camera_name,
                )
                last_eid = None
                for face in faces:
                    last_eid = await self._process_person_face(
                        camera_id, camera_name, frame, det, face
                    )
                return last_eid

            if len(faces) == 1:
                # Single face — full recognition pipeline
                return await self._process_person_face(
                    camera_id, camera_name, frame, det, faces[0]
                )

            # No face detected — try body ReID
            return await self._process_person_no_face(
                camera_id, camera_name, frame, det
            )

        # ─── Non-person: YOLO crop re-verification gate ───
        # When YOLO detects cat/dog, re-run YOLO on the tight crop to confirm
        # the animal is actually there.  Bags, shoes, and cushions that YOLO
        # misclassifies as "cat" on the full frame usually fail re-detection
        # on their own crop.  This gate fires for ALL cat/dog detections,
        # not just matched ones.
        if det.class_name in ("cat", "dog"):
            crop_h, crop_w = det.crop.shape[:2]
            if crop_h >= 50 and crop_w >= 50:
                try:
                    re_dets = await object_detector.detect(
                        det.crop,
                        confidence_threshold=0.25,
                        target_classes=["cat", "dog", "bird", "teddy bear"],
                    )
                    # Accept if crop re-check finds cat/dog directly, OR
                    # finds bird/teddy bear (common YOLO confusion classes for cats)
                    found_animal = any(
                        d.class_name in ("cat", "dog") and d.confidence >= 0.30
                        for d in re_dets
                    )
                    found_confused = any(
                        d.class_name in ("bird", "teddy bear") and d.confidence >= 0.35
                        for d in re_dets
                    )
                    if not found_animal and not found_confused:
                        re_classes = (
                            [(d.class_name, f"{d.confidence:.2f}") for d in re_dets]
                            if re_dets else []
                        )
                        logger.info(
                            "Cat/dog detection REJECTED by crop re-verification: "
                            "camera=%s YOLO=%s conf=%.2f, crop re-check found "
                            "no animal (got: %s)",
                            camera_name, det.class_name, det.confidence, re_classes,
                        )
                        return None
                except Exception as e:
                    logger.debug("Cat/dog re-verification error (non-fatal): %s", e)

        # ─── Non-person: category-isolated embedding match (using cache) ───
        named_object_id = None
        named_object_name = None

        obj_category = EMBEDDING_CATEGORY_MAP.get(det.class_name)
        if obj_category is not None:
            known_objects = self._pet_cache.get(obj_category.value, [])
            if known_objects:
                # Get ALL above-threshold candidates, evaluate each through agent checks
                all_pet_candidates = await recognition_service.match_pet_ranked(
                    det.crop, known_objects,
                )
                for result in all_pet_candidates:
                    # Agent visual sanity check on pet/vehicle match
                    pet_verdict = await recognition_agent.validate_pet_match(
                        det_class=det.class_name,
                        matched_name=result.subject,
                        matched_id=int(result.subject_id),
                        cosine_score=result.confidence,
                        matched_category=obj_category.value,
                        crop=det.crop,
                        det_confidence=det.confidence,
                    )
                    if pet_verdict.accept:
                        named_object_name = result.subject
                        named_object_id = int(result.subject_id)
                        break  # Accepted
                    else:
                        logger.info(
                            "Agent vetoed %s match on %s: %s — trying next candidate",
                            det.class_name, camera_name, pet_verdict.reason,
                        )
                        continue

        # "other" category fallback for unmapped classes
        if named_object_id is None and obj_category is None:
            known_others = self._pet_cache.get(ObjectCategory.other.value, [])
            if known_others:
                all_other_candidates = await recognition_service.match_pet_ranked(
                    det.crop, known_others,
                )
                for result in all_other_candidates:
                    other_verdict = await recognition_agent.validate_pet_match(
                        det_class=det.class_name,
                        matched_name=result.subject,
                        matched_id=int(result.subject_id),
                        cosine_score=result.confidence,
                        matched_category=ObjectCategory.other.value,
                        crop=det.crop,
                        det_confidence=det.confidence,
                    )
                    if other_verdict.accept:
                        named_object_name = result.subject
                        named_object_id = int(result.subject_id)
                        break
                    else:
                        logger.info(
                            "Agent vetoed %s match on %s: %s — trying next candidate",
                            det.class_name, camera_name, other_verdict.reason,
                        )
                        continue

        # Save snapshot + thumbnail
        snapshot_path = await self._save_snapshot(camera_id, frame, det)
        thumbnail_path = await self._save_thumbnail(camera_id, det, frame)

        # Embedding for unnamed objects (clustering) / auto-reinforce for named
        embedding = None
        if named_object_id is None:
            try:
                embedding = await recognition_service.compute_embedding(det.crop)
            except Exception:
                pass
            self._record_static_ghost(camera_id, det)
        else:
            if det.confidence >= 0.65:
                await self._auto_reinforce_embedding(named_object_id, det)
            self._clear_static_ghost(camera_id, det)

        return await self._store_and_notify(
            camera_id, camera_name, frame, det,
            named_object_id, named_object_name,
            snapshot_path, thumbnail_path, embedding,
        )

    # ──────────── Person w/ face detected ────────────

    async def _process_person_face(
        self,
        camera_id: int,
        camera_name: str,
        frame: np.ndarray,
        det: Detection,
        face,
    ):
        """Recognise a person via their detected face (SFace 128-dim embedding)."""
        named_object_id = None
        named_object_name = None

        # Get adaptive recognition plan from the AI agent
        plan = recognition_agent.plan_recognition(camera_id, frame, det.bbox, "person")

        # For multi-face splits, build a sub-detection around this face
        base_x, base_y = det.bbox[0], det.bbox[1]
        fx1 = base_x + face.bbox[0]
        fy1 = base_y + face.bbox[1]
        fx2 = base_x + face.bbox[2]
        fy2 = base_y + face.bbox[3]

        # Expand face bbox to a person-like bbox (some body context)
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        h_frame, w_frame = frame.shape[:2]
        pad = max(face_w, face_h)
        px1 = max(0, fx1 - pad)
        py1 = max(0, fy1 - int(face_h * 0.3))
        px2 = min(w_frame, fx2 + pad)
        py2 = min(h_frame, det.bbox[3])  # Keep bottom of original YOLO box

        person_crop = frame[py1:py2, px1:px2].copy()
        sub_det = Detection("person", det.confidence, (px1, py1, px2, py2), person_crop)

        # Estimate soft biometric attributes (gender/age from face, build/posture from bbox)
        person_attrs = estimate_person_attributes(
            crop=det.crop,
            bbox=det.bbox,
            frame_shape=(h_frame, w_frame),
            face_data=face.face_data,  # InsightFace Face object with gender/age
        )

        # Enhance attributes with ML model when available
        person_attrs = await recognition_agent.enhance_attributes(person_attrs, det.crop, plan)

        # Extract 512-dim face embedding
        if face.embedding and face.face_data is None:
            # Remote ML already computed embedding
            emb = face.embedding
            aligned = face.aligned_crop if face.aligned_crop is not None else np.array([])
        else:
            emb, aligned = await asyncio.to_thread(
                face_service.compute_face_embedding, det.crop, face.face_data
            )

        face_embedding = emb if emb else None
        matched_via_face = False
        ml_attrs = None

        if emb:
            # Match against known person face embeddings (from cache)
            known_persons = self._person_face_cache

            if known_persons:
                # Get ALL above-threshold candidates, evaluate each through agent checks
                all_face_candidates = face_service.match_face_ranked(emb, known_persons)
                ml_attrs = None
                for raw_name, raw_id, raw_score, raw_margin in all_face_candidates:
                    # Apply fused attribute multiplier (heuristic + ML)
                    stored_attrs = self._person_attributes_cache.get(raw_id)
                    if ml_attrs is None:
                        ml_attrs = await recognition_agent.classify_attributes(det.crop) if plan.use_ml_attributes else None
                    attr_mult = recognition_agent.compute_fused_multiplier(person_attrs, ml_attrs, stored_attrs, plan)
                    adjusted_score = raw_score * attr_mult
                    face_threshold = plan.face_threshold
                    # Low margin = ambiguous match → require higher score
                    if raw_margin < 0.05 and len(known_persons) > 2:
                        face_threshold += 0.03
                    # Only accept if adjusted score still passes threshold
                    if adjusted_score < face_threshold:
                        logger.info(
                            "Face match rejected by attributes: %s (raw=%.3f, attr_mult=%.2f, adj=%.3f, thr=%.2f) on %s",
                            raw_name, raw_score, attr_mult, adjusted_score, face_threshold, camera_name,
                        )
                        break  # Score too low — remaining candidates will be even worse
                    # Cross-camera presence check
                    if self.is_presence_conflict(raw_id, camera_id):
                        logger.info(
                            "Face match rejected by presence: %s active on another camera (raw=%.3f) on %s",
                            raw_name, raw_score, camera_name,
                        )
                        continue  # Try next candidate
                    # Agent coherence check — validate match makes sense
                    verdict = recognition_agent.validate_match(
                        camera_id, raw_id, raw_name, raw_score, adjusted_score,
                        person_attrs, stored_attrs, match_method="face",
                        det_class=det.class_name, det_confidence=det.confidence,
                    )
                    if verdict.accept:
                        named_object_name = raw_name
                        named_object_id = raw_id
                        matched_via_face = True
                        logger.info(
                            "Face recognised: %s (raw=%.3f, attr_mult=%.2f, adj=%.3f, thr=%.2f, light=%s) on %s",
                            named_object_name, raw_score, attr_mult, adjusted_score, face_threshold, plan.lighting, camera_name,
                        )
                        break  # Accepted
                    else:
                        logger.info(
                            "Face match vetoed by agent: %s (raw=%.3f, adj=%.3f) — %s on %s — trying next candidate",
                            raw_name, raw_score, adjusted_score, verdict.reason, camera_name,
                        )
                        continue  # Try next candidate

            # Record outcome for agent learning
            recognition_agent.record_outcome(camera_id, "face", success=matched_via_face, face_detected=True)

            # Face didn't match — try body ReID as cross-modal fallback
            if named_object_id is None:
                from services.ml_client import ml_offload_enabled
                if recognition_service.reid_available or ml_offload_enabled:
                    known_body = self._person_body_cache
                    if known_body:
                        body_threshold = plan.body_threshold
                        body_pre = max(0.20, body_threshold - 0.10)
                        # Get ALL above-threshold body candidates, evaluate each
                        all_body_candidates = await recognition_service.match_person_body_ranked(
                            det.crop, known_body, threshold=body_pre,
                        )
                        for body_match in all_body_candidates:
                            body_id = int(body_match.subject_id)
                            # Apply fused attribute multiplier to body match too
                            stored_attrs = self._person_attributes_cache.get(body_id)
                            ml_attrs_body = ml_attrs if ml_attrs else (
                                await recognition_agent.classify_attributes(det.crop) if plan.use_ml_attributes else None
                            )
                            attr_mult = recognition_agent.compute_fused_multiplier(person_attrs, ml_attrs_body, stored_attrs, plan)
                            adjusted_conf = body_match.confidence * attr_mult
                            body_decision_thr = body_threshold
                            if body_match.margin < 0.04 and len(known_body) > 2:
                                body_decision_thr += 0.02
                            if adjusted_conf < body_decision_thr:
                                break  # Score too low — remaining candidates will be worse
                            if self.is_presence_conflict(body_id, camera_id):
                                logger.info(
                                    "Face→body fallback rejected by presence: %s on another camera",
                                    body_match.subject,
                                )
                                continue
                            # Agent coherence check for body match
                            body_verdict = recognition_agent.validate_match(
                                camera_id, body_id, body_match.subject,
                                body_match.confidence, adjusted_conf,
                                person_attrs, stored_attrs, match_method="body",
                                det_class=det.class_name, det_confidence=det.confidence,
                            )
                            if body_verdict.accept:
                                named_object_name = body_match.subject
                                named_object_id = body_id
                                logger.info(
                                    "Face→body fallback: %s (body=%.3f, attr_mult=%.2f, adj=%.3f) on %s",
                                    named_object_name, body_match.confidence, attr_mult, adjusted_conf, camera_name,
                                )
                                # Force-refresh face embedding when body match accepted
                                if emb:
                                    await self._force_refresh_face_embedding(
                                        named_object_id, emb,
                                    )
                                break  # Accepted
                            else:
                                logger.info(
                                    "Face→body fallback vetoed by agent: %s — %s on %s — trying next candidate",
                                    body_match.subject, body_verdict.reason, camera_name,
                                )
                                continue
                        # Record body outcome
                        recognition_agent.record_outcome(
                            camera_id, "body",
                            success=named_object_id is not None,
                        )

        # Learn attributes when person is recognised
        if named_object_id is not None:
            await self._learn_person_attributes(named_object_id, person_attrs)
            recognition_agent.record_appearance(camera_id, named_object_id, person_attrs)

        # Save snapshot and face-focused thumbnail
        snapshot_path = await self._save_snapshot(camera_id, frame, sub_det)
        if aligned is not None and aligned.size > 0:
            thumbnail_path = await self._save_face_thumbnail(camera_id, sub_det, aligned)
        else:
            thumbnail_path = await self._save_thumbnail(camera_id, sub_det, frame)

        # Embedding bookkeeping
        embedding = None
        if named_object_id is None:
            embedding = face_embedding  # Store face embedding for clustering
            self._record_static_ghost(camera_id, det)
        else:
            if face_embedding and matched_via_face:
                await self._auto_reinforce_face_embedding(named_object_id, face_embedding)
            self._clear_static_ghost(camera_id, det)

        # Also reinforce body embedding when person is recognised (any method)
        from services.ml_client import ml_offload_enabled
        if named_object_id is not None and (recognition_service.reid_available or ml_offload_enabled) and det.crop.size > 0:
            body_emb = await recognition_service.compute_reid_embedding_async(det.crop)
            if body_emb:
                await self._auto_reinforce_body_embedding(named_object_id, body_emb)

        return await self._store_and_notify(
            camera_id, camera_name, frame, sub_det,
            named_object_id, named_object_name,
            snapshot_path, thumbnail_path, embedding,
            person_attrs=person_attrs,
        )

    # ──────────── Person w/o face (body ReID) ────────────

    async def _process_person_no_face(
        self,
        camera_id: int,
        camera_name: str,
        frame: np.ndarray,
        det: Detection,
    ):
        """Person detected but no face found — try body-based ReID matching."""
        named_object_id = None
        named_object_name = None
        body_embedding = None

        h_frame, w_frame = frame.shape[:2]

        # Get adaptive recognition plan from the AI agent
        plan = recognition_agent.plan_recognition(camera_id, frame, det.bbox, "person")
        recognition_agent.record_outcome(camera_id, "face", success=False, face_detected=False)

        # Estimate soft biometric attributes (no face data available)
        person_attrs = estimate_person_attributes(
            crop=det.crop,
            bbox=det.bbox,
            frame_shape=(h_frame, w_frame),
            face_data=None,
        )

        # Enhance attributes with ML model (especially valuable without face data)
        person_attrs = await recognition_agent.enhance_attributes(person_attrs, det.crop, plan)

        # ── Stage 1: body ReID against known person body_embeddings ──
        from services.ml_client import ml_offload_enabled
        if recognition_service.reid_available or ml_offload_enabled:
            # Compute body embedding (local or remote)
            if recognition_service.reid_available:
                body_embedding = recognition_service.compute_reid_embedding(det.crop)
            else:
                try:
                    from services.ml_client import remote_embedding
                    body_embedding = await remote_embedding(det.crop, model="reid")
                except Exception:
                    body_embedding = None
            if body_embedding:
                known_persons = self._person_body_cache
                if known_persons:
                    body_threshold = plan.body_threshold
                    body_pre = max(0.20, body_threshold - 0.10)
                    # Get ALL above-threshold body candidates, evaluate each
                    all_body_candidates = await recognition_service.match_person_body_ranked(
                        det.crop, known_persons, threshold=body_pre,
                    )
                    ml_attrs = None
                    for match in all_body_candidates:
                        body_id = int(match.subject_id)
                        # Apply fused attribute multiplier
                        stored_attrs = self._person_attributes_cache.get(body_id)
                        if ml_attrs is None:
                            ml_attrs = await recognition_agent.classify_attributes(det.crop) if plan.use_ml_attributes else None
                        attr_mult = recognition_agent.compute_fused_multiplier(person_attrs, ml_attrs, stored_attrs, plan)
                        adjusted_conf = match.confidence * attr_mult
                        body_decision_thr = body_threshold
                        if match.margin < 0.04 and len(known_persons) > 2:
                            body_decision_thr += 0.02
                        if adjusted_conf < body_decision_thr:
                            break  # Score too low — remaining candidates will be worse
                        if self.is_presence_conflict(body_id, camera_id):
                            logger.info(
                                "Body ReID rejected by presence: %s on another camera",
                                match.subject,
                            )
                            continue
                        # Agent coherence check for body-only match
                        body_verdict = recognition_agent.validate_match(
                            camera_id, body_id, match.subject,
                            match.confidence, adjusted_conf,
                            person_attrs, stored_attrs, match_method="body",
                            det_class=det.class_name, det_confidence=det.confidence,
                        )
                        if body_verdict.accept:
                            named_object_name = match.subject
                            named_object_id = body_id
                            logger.info(
                                "Person via body ReID: %s (body=%.3f, attr_mult=%.2f, adj=%.3f, thr=%.2f, light=%s) on %s",
                                named_object_name, match.confidence, attr_mult, adjusted_conf, body_threshold, plan.lighting, camera_name,
                            )
                            break  # Accepted
                        else:
                            logger.info(
                                "Body ReID vetoed by agent: %s — %s on %s — trying next candidate",
                                match.subject, body_verdict.reason, camera_name,
                            )
                            continue
                    # Record body outcome
                    recognition_agent.record_outcome(camera_id, "body", success=named_object_id is not None)

        # Learn attributes when person is recognised
        if named_object_id is not None:
            await self._learn_person_attributes(named_object_id, person_attrs)
            recognition_agent.record_appearance(camera_id, named_object_id, person_attrs)

        snapshot_path = await self._save_snapshot(camera_id, frame, det)
        thumbnail_path = await self._save_thumbnail(camera_id, det, frame)

        embedding = None
        if named_object_id is None:
            try:
                embedding = await recognition_service.compute_embedding(det.crop)
            except Exception:
                pass
            self._record_static_ghost(camera_id, det)
        else:
            if det.confidence >= 0.65 and body_embedding:
                # Auto-reinforce body embedding for recognised person
                await self._auto_reinforce_body_embedding(named_object_id, body_embedding)
            self._clear_static_ghost(camera_id, det)

        return await self._store_and_notify(
            camera_id, camera_name, frame, det,
            named_object_id, named_object_name,
            snapshot_path, thumbnail_path, embedding,
            person_attrs=person_attrs,
        )

    # ──────────── Shared store + notify ────────────

    async def _store_and_notify(
        self,
        camera_id: int,
        camera_name: str,
        frame: np.ndarray,
        det: Detection,
        named_object_id: Optional[int],
        named_object_name: Optional[str],
        snapshot_path: str,
        thumbnail_path: str,
        embedding: Optional[list] = None,
        person_attrs=None,
    ):
        """Store event in DB, publish to WebSocket, trigger notifications.

        Consolidation: if the same object was recently seen on the same camera
        in roughly the same position, extend the existing event instead of
        creating a new one.
        """
        # Write recognized name back to the live tracker overlay
        if named_object_name:
            object_tracker.set_track_name(camera_id, det.class_name, det.bbox, named_object_name)

        # Update cross-camera presence
        if named_object_id:
            self.update_presence(named_object_id, camera_id)

        event_type = (
            EventType.object_recognized if named_object_id else EventType.object_detected
        )
        meta = {}
        if embedding:
            meta["embedding"] = embedding
        if person_attrs is not None:
            attr_dict = person_attrs.to_dict() if hasattr(person_attrs, 'to_dict') else person_attrs
            if attr_dict:
                meta["attributes"] = attr_dict
        annotations = self._collect_event_annotations(camera_id, det, named_object_name)
        if annotations:
            meta["annotations"] = annotations

        # Generate activity narrative
        posture = None
        if person_attrs is not None:
            attr_dict_for_posture = person_attrs.to_dict() if hasattr(person_attrs, 'to_dict') else person_attrs
            if attr_dict_for_posture:
                posture = attr_dict_for_posture.get("posture")
        narrative = generate_narrative(
            named_object_name=named_object_name,
            object_type=det.class_name,
            camera_name=camera_name,
            timestamp=datetime.now(timezone.utc),
            posture=posture,
        )
        meta["narrative"] = narrative

        # Store detection resolution for bbox scaling when snapshot is re-annotated
        meta["detect_resolution"] = [frame.shape[1], frame.shape[0]]

        bbox_dict = det.to_dict()["bbox"]

        # ── Try to consolidate with a recent event ──
        existing_event_id = self._try_consolidate(
            camera_id, det.class_name, named_object_id, bbox_dict
        )

        if existing_event_id:
            # Extend the existing event — update ended_at, bump snapshot
            build_gif = False
            async with async_session() as session:
                event = await session.get(Event, existing_event_id)
                if event:
                    event.ended_at = datetime.now(timezone.utc)
                    # Update snapshot/thumbnail to latest
                    event.snapshot_path = snapshot_path
                    old_thumb = event.thumbnail_path
                    event.thumbnail_path = thumbnail_path
                    event.bbox = bbox_dict
                    if event.confidence and det.confidence > event.confidence:
                        event.confidence = det.confidence
                    # Upgrade from detected → recognized if we now have a match
                    if named_object_id and not event.named_object_id:
                        event.named_object_id = named_object_id
                        event.event_type = EventType.object_recognized
                    # Merge metadata
                    existing_meta = event.metadata_extra or {}
                    if meta.get("attributes"):
                        existing_meta["attributes"] = meta["attributes"]
                    if meta.get("embedding") and "embedding" not in existing_meta:
                        existing_meta["embedding"] = meta["embedding"]
                    if meta.get("annotations"):
                        existing_meta["annotations"] = meta["annotations"]
                    if meta.get("narrative"):
                        existing_meta["narrative"] = meta["narrative"]
                    existing_meta["consolidated"] = existing_meta.get("consolidated", 1) + 1

                    # ── Accumulate thumbnail history for preview GIF ──
                    thumb_history = existing_meta.get("thumbnail_history", [])
                    # Seed with old thumbnail if this is the first consolidation
                    if not thumb_history and old_thumb:
                        thumb_history.append(old_thumb)
                    if thumbnail_path and thumbnail_path not in thumb_history:
                        thumb_history.append(thumbnail_path)
                    # Cap to most recent N frames
                    if len(thumb_history) > self._MAX_CONSOLIDATION_FRAMES:
                        thumb_history = thumb_history[-self._MAX_CONSOLIDATION_FRAMES:]
                    existing_meta["thumbnail_history"] = thumb_history
                    build_gif = len(thumb_history) >= 2

                    event.metadata_extra = existing_meta
                    await session.commit()
                    event_id = existing_event_id
                    cam_id_for_gif = event.camera_id
                else:
                    existing_event_id = None  # Event was deleted — fall through

            if existing_event_id:
                # Build preview GIF from accumulated thumbnails (non-blocking)
                if build_gif:
                    try:
                        gif_path = await self._build_consolidation_gif(
                            cam_id_for_gif, thumb_history, det.class_name,
                        )
                        if gif_path:
                            async with async_session() as session:
                                ev = await session.get(Event, event_id)
                                if ev:
                                    m = ev.metadata_extra or {}
                                    # Remove old GIF file if different
                                    old_gif = m.get("gif_path")
                                    if old_gif and old_gif != gif_path:
                                        try:
                                            Path(old_gif).unlink(missing_ok=True)
                                        except Exception:
                                            pass
                                    m["gif_path"] = gif_path
                                    ev.metadata_extra = m
                                    await session.commit()
                    except Exception as e:
                        logger.warning("Consolidation GIF failed: %s", e)

                # Record the extension
                self._record_event(camera_id, det.class_name, named_object_id, event_id, bbox_dict)

                # Publish a lightweight update (not a new detection)
                await event_bus.publish({
                    "type": "event_updated",
                    "event_id": event_id,
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "object_type": det.class_name,
                    "named_object": named_object_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                logger.debug(
                    "Consolidated: camera=%s type=%s name=%s → event %d",
                    camera_name, det.class_name, named_object_name or "unknown", event_id,
                )
                return event_id

        # ── Create new event ──
        async with async_session() as session:
            event = Event(
                camera_id=camera_id,
                event_type=event_type,
                object_type=det.class_name,
                named_object_id=named_object_id,
                confidence=det.confidence,
                snapshot_path=snapshot_path,
                thumbnail_path=thumbnail_path,
                bbox=bbox_dict,
                started_at=datetime.now(timezone.utc),
                metadata_extra=meta if meta else None,
            )
            session.add(event)
            await session.commit()
            await session.refresh(event)
            event_id = event.id

        # Record for future consolidation
        self._record_event(camera_id, det.class_name, named_object_id, event_id, bbox_dict)

        ws_event = {
            "type": "detection",
            "event_id": event_id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "object_type": det.class_name,
            "named_object": named_object_name,
            "confidence": round(det.confidence, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot_url": f"/api/events/{event_id}/snapshot",
        }
        await event_bus.publish(ws_event)

        if self._notification_engine:
            # Suppress notification if this named object just re-appeared after a
            # brief tracking gap (track cycling) — the tracker's identity memory
            # proves the object never really left.
            if named_object_name and object_tracker.is_named_object_recently_departed(
                camera_id, named_object_name
            ):
                logger.info(
                    "Skipping notification for re-identified %s on %s (recently departed)",
                    named_object_name, camera_name,
                )
            else:
                await self._notification_engine.evaluate_and_notify(
                    camera_id=camera_id,
                    camera_name=camera_name,
                    object_type=det.class_name,
                    named_object_id=named_object_id,
                    named_object_name=named_object_name,
                    event_id=event_id,
                    snapshot_path=snapshot_path,
                )

        logger.info(
            "Detection: camera=%s type=%s name=%s conf=%.2f",
            camera_name, det.class_name, named_object_name or "unknown", det.confidence,
        )
        return event_id

    async def _store_motion_event(self, camera_id: int, camera_name: str, frame: np.ndarray, motion_score: float = 0.0, motion_regions: list | None = None):
        """Store a motion event with snapshot and trigger notifications."""
        # Save snapshot with motion regions highlighted
        snapshot_path = await self._save_motion_snapshot(camera_id, frame, motion_regions)

        async with async_session() as session:
            event = Event(
                camera_id=camera_id,
                event_type=EventType.motion,
                object_type="motion",
                snapshot_path=snapshot_path,
                started_at=datetime.now(timezone.utc),
                metadata_extra={"motion_score": motion_score, "motion_regions": motion_regions or []},
            )
            session.add(event)
            await session.commit()
            await session.refresh(event)
            event_id = event.id

        await event_bus.publish({
            "type": "motion",
            "event_id": event_id,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "motion_score": motion_score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "snapshot_url": f"/api/events/{event_id}/snapshot",
        })

        # Only notify for significant motion (>2% of frame area)
        if motion_score >= 0.02 and self._notification_engine:
            await self._notification_engine.evaluate_and_notify(
                camera_id=camera_id,
                camera_name=camera_name,
                object_type="motion",
                named_object_id=None,
                named_object_name=None,
                event_id=event_id,
                snapshot_path=snapshot_path,
            )

    async def _auto_reinforce_embedding(self, named_object_id: int, det: Detection):
        """Continuously reinforce a named object's embedding from high-confidence detections."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(NamedObject).where(NamedObject.id == named_object_id)
                )
                obj = result.scalar_one_or_none()
                if not obj or obj.reference_image_count >= 50:
                    return

                raw_emb = await recognition_service.compute_embedding(det.crop)
                # Similarity gate: reject if new embedding diverges from stored
                if obj.embedding and len(obj.embedding) == len(raw_emb):
                    old = np.array(obj.embedding, dtype=np.float64)
                    new = np.array(raw_emb, dtype=np.float64)
                    sim = float(np.dot(old, new) / (np.linalg.norm(old) * np.linalg.norm(new) + 1e-10))
                    if sim < 0.55:
                        logger.debug("CNN reinforce skipped for %s: sim=%.3f", obj.name, sim)
                        return
                    merged = (old * obj.reference_image_count + new) / (obj.reference_image_count + 1)
                    norm = np.linalg.norm(merged)
                    if norm > 0:
                        merged = merged / norm
                    obj.embedding = merged.tolist()
                else:
                    obj.embedding = raw_emb
                obj.reference_image_count += 1
                session.add(obj)
                await session.commit()
        except Exception as e:
            logger.debug("Auto-reinforce embedding failed for object %d: %s", named_object_id, e)

    async def _auto_reinforce_face_embedding(
        self, named_object_id: int, face_embedding: list[float]
    ):
        """Reinforce a person's face embedding from a high-confidence face detection."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(NamedObject).where(NamedObject.id == named_object_id)
                )
                obj = result.scalar_one_or_none()
                if not obj or obj.reference_image_count >= 50:
                    return

                # Similarity gate: reject if face embedding diverges from stored
                if obj.embedding and len(obj.embedding) == len(face_embedding):
                    existing = np.array(obj.embedding, dtype=np.float64)
                    candidate = np.array(face_embedding, dtype=np.float64)
                    sim = float(np.dot(existing, candidate) / (np.linalg.norm(existing) * np.linalg.norm(candidate) + 1e-10))
                    if sim < 0.30:
                        logger.debug("Face reinforce skipped for %s: sim=%.3f", obj.name, sim)
                        return

                obj.embedding = face_service.merge_face_embeddings(
                    obj.embedding, face_embedding, obj.reference_image_count
                )
                obj.reference_image_count += 1
                session.add(obj)
                await session.commit()
        except Exception as e:
            logger.debug("Face embedding reinforce failed for object %d: %s", named_object_id, e)

    async def _force_refresh_face_embedding(
        self, named_object_id: int, face_embedding: list[float]
    ):
        """Replace a person's face embedding entirely (model change detected via body fallback)."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(NamedObject).where(NamedObject.id == named_object_id)
                )
                obj = result.scalar_one_or_none()
                if not obj:
                    return

                import numpy as np
                new_arr = np.array(face_embedding, dtype=np.float64)
                norm = np.linalg.norm(new_arr)
                if norm > 0:
                    new_arr = new_arr / norm
                obj.embedding = new_arr.tolist()
                obj.reference_image_count = 1
                logger.info(
                    "Force-refreshed face embedding for %s (dim=%d, was %d)",
                    obj.name, len(face_embedding),
                    len(obj.embedding) if obj.embedding else 0,
                )
                session.add(obj)
                await session.commit()
        except Exception as e:
            logger.debug("Force face refresh failed for object %d: %s", named_object_id, e)

    async def _auto_reinforce_body_embedding(
        self, named_object_id: int, body_embedding: list[float]
    ):
        """Reinforce a person's body ReID embedding from a recognition event."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(NamedObject).where(NamedObject.id == named_object_id)
                )
                obj = result.scalar_one_or_none()
                if not obj or obj.reference_image_count >= 50:
                    return

                # Similarity gate: reject if body embedding diverges from stored
                if obj.body_embedding and len(obj.body_embedding) == len(body_embedding):
                    existing = np.array(obj.body_embedding, dtype=np.float64)
                    candidate = np.array(body_embedding, dtype=np.float64)
                    sim = float(np.dot(existing, candidate) / (np.linalg.norm(existing) * np.linalg.norm(candidate) + 1e-10))
                    if sim < 0.55:
                        logger.debug("Body reinforce skipped for %s: sim=%.3f", obj.name, sim)
                        return

                obj.body_embedding = recognition_service.merge_reid_embedding(
                    obj.body_embedding, body_embedding, obj.reference_image_count
                )
                session.add(obj)
                await session.commit()
        except Exception as e:
            logger.debug("Body embedding reinforce failed for object %d: %s", named_object_id, e)

    async def _learn_person_attributes(self, named_object_id: int, attrs):
        """Merge detected soft biometric attributes into a named person's profile."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(NamedObject).where(NamedObject.id == named_object_id)
                )
                obj = result.scalar_one_or_none()
                if not obj:
                    return
                updated = merge_stable_attributes(obj.attributes, attrs)
                obj.attributes = updated
                session.add(obj)
                await session.commit()
                # Update in-memory cache
                self._person_attributes_cache[named_object_id] = updated
        except Exception as e:
            logger.debug("Attribute learning failed for object %d: %s", named_object_id, e)

    # ──────────── Recognition retry (no event creation) ────────────

    async def _try_recognize(
        self, camera_id: int, camera_name: str, frame: np.ndarray, det: Detection
    ) -> Optional[tuple]:
        """Try to identify a detection without creating an event.

        Returns (named_object_id, named_object_name, embedding_list) on match,
        or None if unrecognised.
        """
        await self._refresh_embedding_cache()

        EMBEDDING_CATEGORY_MAP = {
            "cat": ObjectCategory.pet, "dog": ObjectCategory.pet,
            "car": ObjectCategory.vehicle, "truck": ObjectCategory.vehicle,
            "bus": ObjectCategory.vehicle, "motorcycle": ObjectCategory.vehicle,
            "bicycle": ObjectCategory.vehicle, "boat": ObjectCategory.vehicle,
        }

        if det.class_name == "person":
            h_frame, w_frame = frame.shape[:2]

            # Get adaptive plan from the recognition agent
            plan = recognition_agent.plan_recognition(camera_id, frame, det.bbox, "person")

            person_attrs = estimate_person_attributes(
                crop=det.crop, bbox=det.bbox,
                frame_shape=(h_frame, w_frame), face_data=None,
            )

            # Enhance attributes with ML model
            person_attrs = await recognition_agent.enhance_attributes(person_attrs, det.crop, plan)

            # Try face recognition (with upscale retry for small crops)
            faces = await face_service.detect_faces_async(det.crop)
            if not faces:
                h, w = det.crop.shape[:2]
                if h < 200 or w < 120:
                    scale = min(2.5, max(200 / h, 120 / w))
                    upscaled = cv2.resize(det.crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    up_faces = await face_service.detect_faces_async(upscaled)
                    if up_faces:
                        for f in up_faces:
                            bx1, by1, bx2, by2 = f.bbox
                            f.bbox = (int(bx1 / scale), int(by1 / scale), int(bx2 / scale), int(by2 / scale))
                        faces = up_faces

            if faces:
                face = faces[0]
                # Update attributes with face data if available
                if face.face_data is not None:
                    person_attrs = estimate_person_attributes(
                        crop=det.crop, bbox=det.bbox,
                        frame_shape=(h_frame, w_frame), face_data=face.face_data,
                    )
                if face.embedding and face.face_data is None:
                    emb = face.embedding
                else:
                    emb, _ = await asyncio.to_thread(
                        face_service.compute_face_embedding, det.crop, face.face_data
                    )
                if emb:
                    known_persons = self._person_face_cache
                    if known_persons:
                        # Get ALL above-threshold face candidates, evaluate each
                        all_face_candidates = face_service.match_face_ranked(emb, known_persons)
                        ml_attrs = None
                        for name, obj_id, score, margin in all_face_candidates:
                            stored_attrs = self._person_attributes_cache.get(obj_id)
                            if ml_attrs is None:
                                ml_attrs = await recognition_agent.classify_attributes(det.crop) if plan.use_ml_attributes else None
                            attr_mult = recognition_agent.compute_fused_multiplier(person_attrs, ml_attrs, stored_attrs, plan)
                            face_threshold = plan.face_threshold
                            if margin < 0.05 and len(known_persons) > 2:
                                face_threshold += 0.03
                            if score * attr_mult < face_threshold:
                                break  # Score too low — remaining candidates will be worse
                            if self.is_presence_conflict(obj_id, camera_id):
                                continue
                            verdict = recognition_agent.validate_match(
                                camera_id, obj_id, name, score, score * attr_mult,
                                person_attrs, stored_attrs, match_method="face",
                                det_class=det.class_name, det_confidence=det.confidence,
                            )
                            if verdict.accept:
                                recognition_agent.record_outcome(camera_id, "face", success=True, face_detected=True)
                                await self._learn_person_attributes(obj_id, person_attrs)
                                recognition_agent.record_appearance(camera_id, obj_id, person_attrs)
                                self.update_presence(obj_id, camera_id)
                                return (obj_id, name, emb)
                            continue
                    recognition_agent.record_outcome(camera_id, "face", success=False, face_detected=True)
            else:
                recognition_agent.record_outcome(camera_id, "face", success=False, face_detected=False)

            # Face failed — try body ReID
            from services.ml_client import ml_offload_enabled
            if recognition_service.reid_available or ml_offload_enabled:
                known_body = self._person_body_cache
                if known_body:
                    body_threshold = plan.body_threshold
                    body_pre = max(0.20, body_threshold - 0.10)
                    # Get ALL above-threshold body candidates, evaluate each
                    all_body_candidates = await recognition_service.match_person_body_ranked(
                        det.crop, known_body, threshold=body_pre,
                    )
                    ml_attrs = None
                    for body_match in all_body_candidates:
                        body_id = int(body_match.subject_id)
                        stored_attrs = self._person_attributes_cache.get(body_id)
                        if ml_attrs is None:
                            ml_attrs = await recognition_agent.classify_attributes(det.crop) if plan.use_ml_attributes else None
                        attr_mult = recognition_agent.compute_fused_multiplier(person_attrs, ml_attrs, stored_attrs, plan)
                        body_decision_thr = body_threshold
                        if body_match.margin < 0.04 and len(known_body) > 2:
                            body_decision_thr += 0.02
                        if body_match.confidence * attr_mult < body_decision_thr:
                            break  # Score too low — remaining candidates will be worse
                        if self.is_presence_conflict(body_id, camera_id):
                            continue
                        body_verdict = recognition_agent.validate_match(
                            camera_id, body_id, body_match.subject,
                            body_match.confidence, body_match.confidence * attr_mult,
                            person_attrs, stored_attrs, match_method="body",
                            det_class=det.class_name, det_confidence=det.confidence,
                        )
                        if body_verdict.accept:
                            recognition_agent.record_outcome(camera_id, "body", success=True)
                            await self._learn_person_attributes(body_id, person_attrs)
                            recognition_agent.record_appearance(camera_id, body_id, person_attrs)
                            self.update_presence(body_id, camera_id)
                            return (body_id, body_match.subject, None)
                        continue
                    recognition_agent.record_outcome(camera_id, "body", success=False)
            return None

        # Non-person: CNN/pet embedding match (from cache)
        obj_category = EMBEDDING_CATEGORY_MAP.get(det.class_name)
        if obj_category is not None:
            known_objects = self._pet_cache.get(obj_category.value, [])
            if known_objects:
                # Get ALL above-threshold candidates, evaluate each
                all_pet_candidates = await recognition_service.match_pet_ranked(
                    det.crop, known_objects,
                )
                for result in all_pet_candidates:
                    pet_verdict = await recognition_agent.validate_pet_match(
                        det_class=det.class_name,
                        matched_name=result.subject,
                        matched_id=int(result.subject_id),
                        cosine_score=result.confidence,
                        matched_category=obj_category.value,
                        crop=det.crop,
                        det_confidence=det.confidence,
                    )
                    if pet_verdict.accept:
                        return (int(result.subject_id), result.subject, None)
                    continue
        return None

    # ──────────── GIF timelapse builders ────────────

    # Max thumbnail frames to keep in consolidation history
    _MAX_CONSOLIDATION_FRAMES = 12

    async def _build_consolidation_gif(
        self, camera_id: int, thumbnail_paths: list[str], class_name: str,
    ) -> Optional[str]:
        """Build an animated GIF from accumulated consolidation thumbnails.

        Reads JPEG thumbnails from disk, resizes to uniform dimensions,
        and saves as a looping GIF.  Returns the file path or None.
        """
        if len(thumbnail_paths) < 2:
            return None
        try:
            from PIL import Image

            gif_dir = Path(_hot_snapshots_path()) / str(camera_id)
            gif_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{class_name}_preview.gif"
            filepath = gif_dir / filename

            def _save():
                frames: list[Image.Image] = []
                for tp in thumbnail_paths:
                    p = Path(tp)
                    if not p.exists():
                        # Try hot/cold flip
                        alt = str(tp).replace("/recordings/", "/mnt/ssd/") if "/recordings/" in tp else str(tp).replace("/mnt/ssd/", "/recordings/")
                        p = Path(alt)
                        if not p.exists():
                            continue
                    img = Image.open(p).convert("RGB")
                    frames.append(img)
                if len(frames) < 2:
                    return False
                # Uniform size based on first frame
                tw, th = frames[0].size
                resized = []
                for f in frames:
                    if f.size != (tw, th):
                        f = f.resize((tw, th), Image.LANCZOS)
                    resized.append(f)
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
                logger.debug("Built consolidation GIF: %s (%d frames)", filepath, len(thumbnail_paths))
                return str(filepath)
            return None
        except Exception as e:
            logger.error("Failed to build consolidation GIF for camera %d: %s", camera_id, e)
            return None

    async def _build_gif(
        self, camera_id: int, crops: list[np.ndarray], class_name: str
    ) -> Optional[str]:
        """Build an animated GIF from collected tracking crops.

        Returns the file path of the saved GIF, or None on failure.
        """
        try:
            from PIL import Image

            gif_dir = Path(_hot_snapshots_path()) / str(camera_id)
            gif_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{class_name}_timelapse.gif"
            filepath = gif_dir / filename

            # Determine uniform size (use first crop dimensions)
            target_h, target_w = crops[0].shape[:2]

            def _save():
                frames = []
                for crop in crops:
                    # Resize to uniform dimensions
                    h, w = crop.shape[:2]
                    if (w, h) != (target_w, target_h):
                        crop = cv2.resize(crop, (target_w, target_h))
                    # BGR → RGB for PIL
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                # Save animated GIF: 500ms per frame, loop forever
                frames[0].save(
                    str(filepath),
                    save_all=True,
                    append_images=frames[1:],
                    duration=500,
                    loop=0,
                    optimize=True,
                )

            await asyncio.to_thread(_save)
            logger.info("Built GIF timelapse: %s (%d frames)", filepath, len(crops))
            return str(filepath)
        except Exception as e:
            logger.error("Failed to build GIF for camera %d: %s", camera_id, e)
            return None

    async def _save_motion_snapshot(self, camera_id: int, frame: np.ndarray, motion_regions: list | None = None) -> str:
        """Save a snapshot with motion regions highlighted in red."""
        snap_dir = Path(_hot_snapshots_path()) / str(camera_id)
        snap_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_motion.jpg"
        filepath = snap_dir / filename

        if motion_regions:
            annotated = frame.copy()
            for x, y, w, h in motion_regions:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, "MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            await asyncio.to_thread(cv2.imwrite, str(filepath), annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        else:
            await asyncio.to_thread(cv2.imwrite, str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return str(filepath)

    async def _save_snapshot(self, camera_id: int, frame: np.ndarray, det: Detection) -> str:
        """Save full frame with bounding box overlay."""
        snap_dir = Path(_hot_snapshots_path()) / str(camera_id)
        snap_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{det.class_name}.jpg"
        filepath = snap_dir / filename

        # Save clean frame first (no annotations) for face re-detection
        clean_filename = f"{ts}_{det.class_name}_clean.jpg"
        clean_path = snap_dir / clean_filename
        await asyncio.to_thread(cv2.imwrite, str(clean_path), frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])

        # Draw bounding box on copy
        annotated = frame.copy()
        x1, y1, x2, y2 = det.bbox
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        await asyncio.to_thread(cv2.imwrite, str(filepath), annotated, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return str(filepath)

    @staticmethod
    def compute_centered_crop(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        class_name: str = "person",
        padding_ratio: float = 0.5,
    ) -> np.ndarray:
        """Compute a generously-padded, subject-centered crop from the full frame.

        For thumbnails used in manual confirmation, this produces a much wider
        view than the tight YOLO bbox, ensuring the whole person (or object)
        is visible with surrounding context.

        Args:
            frame: Full camera frame (H, W, 3).
            bbox:  Original detection bbox (x1, y1, x2, y2).
            class_name: Detection class (person gets more padding).
            padding_ratio: Fraction of bbox dimension to add on each side.
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # Tight padding so the subject fills the thumbnail. Small head margin
        # for people; uniform tight margin for everything else. Wider crops
        # waste pixels on background and confuse downstream vision models.
        if class_name == "person":
            pad_top = int(box_h * 0.15)
            pad_bottom = int(box_h * 0.05)
            pad_left = int(box_w * 0.08)
            pad_right = int(box_w * 0.08)
        elif class_name in ("cat", "dog"):
            pad = int(max(box_w, box_h) * 0.10)
            pad_top = pad_bottom = pad_left = pad_right = pad
        else:
            pad = int(max(box_w, box_h) * 0.12)
            pad_top = pad_bottom = pad_left = pad_right = pad

        # Desired crop region (may exceed frame)
        cx1 = x1 - pad_left
        cy1 = y1 - pad_top
        cx2 = x2 + pad_right
        cy2 = y2 + pad_bottom

        # Clamp to frame, but try to keep subject centred by shifting
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # Shift horizontally if we hit frame edges
        if cx1 < 0:
            cx2 = min(fw, cx2 - cx1)
            cx1 = 0
        if cx2 > fw:
            cx1 = max(0, cx1 - (cx2 - fw))
            cx2 = fw
        # Shift vertically
        if cy1 < 0:
            cy2 = min(fh, cy2 - cy1)
            cy1 = 0
        if cy2 > fh:
            cy1 = max(0, cy1 - (cy2 - fh))
            cy2 = fh

        # Final clamp
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(fw, cx2), min(fh, cy2)

        if cx2 <= cx1 or cy2 <= cy1:
            # Fallback to raw bbox
            return frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)].copy()

        return frame[cy1:cy2, cx1:cx2].copy()

    async def _save_thumbnail(self, camera_id: int, det: Detection, frame: np.ndarray = None) -> str:
        """Save cropped object thumbnail.

        When *frame* is provided, produces a wider centred crop for better
        manual-confirmation UX.  Falls back to det.crop when frame is absent.
        """
        thumb_dir = Path(_hot_snapshots_path()) / str(camera_id) / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{det.class_name}_thumb.jpg"
        filepath = thumb_dir / filename

        # Use centred crop from full frame when available
        if frame is not None and det.bbox:
            crop = self.compute_centered_crop(frame, det.bbox, det.class_name)
        else:
            crop = det.crop

        # Person crops: 480px wide for reliable face detection during retraining
        # Other objects: 200px wide (sufficient for CNN matching)
        max_width = 480 if det.class_name == "person" else 200
        h, w = crop.shape[:2]
        if w > max_width:
            scale = max_width / w
            crop = cv2.resize(crop, (max_width, int(h * scale)))

        await asyncio.to_thread(cv2.imwrite, str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return str(filepath)

    async def _save_face_thumbnail(
        self, camera_id: int, det: Detection, face_crop: np.ndarray
    ) -> str:
        """Save an aligned-face thumbnail (112×112 from SFace)."""
        thumb_dir = Path(_hot_snapshots_path()) / str(camera_id) / "thumbs"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{det.class_name}_face_thumb.jpg"
        filepath = thumb_dir / filename

        await asyncio.to_thread(cv2.imwrite, str(filepath), face_crop, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        return str(filepath)


# Singleton
event_processor = EventProcessor()
