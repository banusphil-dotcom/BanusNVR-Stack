"""BanusNas — Object Tracker: BoT-SORT + ByteTrack with ReID + optical flow.

Upgraded tracking pipeline:
  - BoT-SORT Global Motion Compensation (GMC) via sparse optical flow + RANSAC
  - Inter-detection optical flow for sub-second position updates
  - Mask IoU when segmentation model is available
  - Person ReID (OpenVINO 256-dim) for appearance-enhanced association
  - ByteTrack two-pass high/low confidence association
"""

import asyncio
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import av
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from services.stream_manager import stream_manager
from services.object_detector import object_detector, Detection

logger = logging.getLogger(__name__)

# Adaptive: run YOLO more frequently when objects are active
TRACK_INTERVAL_ACTIVE = 1.5   # When tracks exist
# Seconds without seeing a tracked object before it's considered departed
TRACK_LOST_TIMEOUT = 30.0

# ── Motion-gated detection ──
# Only run YOLO when motion is detected or tracks need verification.
# Saves ~80% of GPU/Coral load on quiet cameras.
MOTION_THRESHOLD = 0.003       # Min foreground ratio to count as motion (0.3% of frame)
MOTION_FRAMES_REQUIRED = 2     # Consecutive motion frames before triggering detection
MOTION_COOLDOWN = 1.0          # Min seconds between motion triggers
STATIONARY_VERIFY_INTERVAL = 30.0  # Re-run YOLO on stationary tracks every N seconds
MOTION_DETECT_INTERVAL = 0.3   # Check for motion every N seconds (cheap frame diff)
# Minimum consecutive detections before confirming a new track
TRACK_CONFIRM_FRAMES = 2
# Seconds between recognition retries for unnamed tracked objects
SNAPSHOT_INTERVAL = 60.0
# GIF/timelapse crop collection
GIF_CROP_INTERVAL = 15.0       # Seconds between GIF frame captures
MAX_GIF_CROPS = 40             # Max frames per track (~10 min at 15s)

# ByteTrack thresholds
BYTETRACK_HIGH_THRESH = 0.3   # High-confidence detection threshold for first association
BYTETRACK_LOW_THRESH = 0.1    # Low-confidence threshold for second association
BYTETRACK_IOU_THRESH = 0.25   # IoU threshold for matching
BYTETRACK_SECOND_IOU = 0.35   # IoU for second-pass matching

# Seconds to remember departed named identities for re-identification
IDENTITY_MEMORY_TIMEOUT = 120.0
# IoU threshold for matching a new track to a remembered identity
IDENTITY_IOU_THRESHOLD = 0.25
# Seconds to remember departed pet identities (longer — cats disappear/reappear)
PET_IDENTITY_MEMORY_TIMEOUT = 300.0

# ── BoT-SORT: Global Motion Compensation (GMC) ──
GMC_MAX_CORNERS = 200       # Shi-Tomasi corners for camera motion estimation
GMC_QUALITY_LEVEL = 0.01    # Corner quality threshold
GMC_MIN_DISTANCE = 30       # Min pixel distance between corners

# ── Inter-detection optical flow ──
FLOW_INTERVAL = 0.5          # Seconds between LK flow updates
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# ── StrongSORT appearance-enhanced association ──
# Weight for IoU cost vs appearance cost in person association (α for IoU)
APPEARANCE_ALPHA = 0.6  # 0.6 * IoU + 0.4 * appearance
# Cosine distance gate: ignore appearance matches below this similarity
APPEARANCE_GATE = 0.20
# EMA smoothing for track appearance embeddings
APPEARANCE_EMA = 0.9  # track_emb = 0.9 * old + 0.1 * new
# Classes that use appearance-enhanced tracking
APPEARANCE_CLASSES = {"person"}

# Enhanced detection for small objects (cats, dogs)
ENHANCED_SCAN_INTERVAL = 10.0  # seconds between enhanced scans (idle/discovery)
ENHANCED_SCAN_ACTIVE = 3.0     # seconds between enhanced scans when sustaining tracks
SMALL_OBJECT_CLASSES = {"cat", "dog", "bird"}

# Per-class lost timeout overrides (seconds without detection before departure)
CLASS_LOST_TIMEOUT = {
    "cat": 60.0,   # Cats are hard to detect continuously — longer timeout
    "dog": 60.0,
}

# Per-class Kalman noise tuning — small/slow animals need less velocity noise
CLASS_KALMAN_PROCESS_NOISE = {
    "cat": 3.0,
    "dog": 3.0,
    "person": 10.0,
}


class BootstrapScheduler:
    """Coordinate first-inference bootstrapping across cameras.

    This prevents all trackers from trying to run their first YOLO pass at the
    same moment during app startup or service recovery.
    """

    def __init__(self, max_concurrent: int = 1):
        self._semaphore = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._max_concurrent = max_concurrent
        self._active: set[int] = set()
        self._waits = 0
        self._grants = 0

    def try_acquire(self, camera_id: int) -> bool:
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            with self._lock:
                self._waits += 1
            return False
        with self._lock:
            self._active.add(camera_id)
            self._grants += 1
        return True

    def release(self, camera_id: int):
        with self._lock:
            if camera_id not in self._active:
                return
            self._active.remove(camera_id)
        self._semaphore.release()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "max_concurrent": self._max_concurrent,
                "active": len(self._active),
                "active_camera_ids": sorted(self._active),
                "waits": self._waits,
                "grants": self._grants,
            }


BOOTSTRAP_SCHEDULER = BootstrapScheduler(max_concurrent=1)


class KalmanBoxTracker:
    """Simple Kalman filter for bounding box tracking (constant velocity model)."""

    _count = 0

    def __init__(self, bbox: tuple[int, int, int, int], class_name: str = "person"):
        from cv2 import KalmanFilter as KF
        # State: [cx, cy, s, r, dcx, dcy, ds] where s=area, r=aspect ratio
        self.kf = KF(7, 4, 0)
        self.kf.measurementMatrix = np.eye(4, 7, dtype=np.float32)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        # Per-class process noise: less for small/slow animals, more for fast people
        pn_scale = CLASS_KALMAN_PROCESS_NOISE.get(class_name, 10.0)
        self.kf.processNoiseCov *= pn_scale
        self.kf.processNoiseCov[4:, 4:] *= 10.0  # velocity terms
        self.kf.measurementNoiseCov *= 1.0

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        self.kf.statePost = np.array([[cx], [cy], [s], [r], [0], [0], [0]], dtype=np.float32)
        KalmanBoxTracker._count += 1

    def predict(self) -> tuple[int, int, int, int]:
        if self.kf.statePost[2, 0] + self.kf.statePost[6, 0] <= 0:
            self.kf.statePost[6, 0] = 0.0
        self.kf.predict()
        return self._state_to_bbox()

    def update(self, bbox: tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        s = w * h
        r = w / max(h, 1e-6)
        self.kf.correct(np.array([[cx], [cy], [s], [r]], dtype=np.float32))

    def _state_to_bbox(self) -> tuple[int, int, int, int]:
        cx = self.kf.statePost[0, 0]
        cy = self.kf.statePost[1, 0]
        s = max(self.kf.statePost[2, 0], 1.0)
        r = max(self.kf.statePost[3, 0], 0.01)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    @property
    def predicted_bbox(self) -> tuple[int, int, int, int]:
        return self._state_to_bbox()


@dataclass
class TrackedObject:
    """A single tracked object across frames — with Kalman-predicted motion."""

    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    confidence: float
    crop: np.ndarray
    first_seen: float
    last_seen: float
    kalman: KalmanBoxTracker = field(repr=False, default=None)
    consecutive: int = 1
    confirmed: bool = False
    # Set to True once appear event has been fired
    appear_notified: bool = False
    last_snapshot_time: float = 0.0
    named_object_name: str | None = None
    # StrongSORT: smoothed ReID embedding for appearance-enhanced association
    reid_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # Segmentation mask (bbox-cropped binary, optional)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    # Event lifecycle: DB event_id set after appear callback
    event_id: Optional[int] = None
    # GIF/timelapse: collected crops during track lifetime
    gif_crops: list = field(default_factory=list, repr=False)
    last_gif_crop_time: float = 0.0

    def __post_init__(self):
        if self.kalman is None:
            self.kalman = KalmanBoxTracker(self.bbox, self.class_name)


@dataclass
class DepartedIdentity:
    """Memory of a recently-departed named track for re-identification."""

    class_name: str
    bbox: tuple[int, int, int, int]
    named_object_name: str
    departed_at: float
    # StrongSORT: appearance embedding for appearance-based re-ID
    reid_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # CNN embedding for pet/object re-ID (MobileNetV2 1280-dim)
    cnn_embedding: Optional[np.ndarray] = field(default=None, repr=False)
    # Event continuity: carry event_id and gif_crops so re-acquired tracks
    # continue the same event rather than creating a new one
    event_id: Optional[int] = None
    gif_crops: list = field(default_factory=list, repr=False)
    first_seen: float = 0.0


class CameraObjectTracker:
    """Continuously pulls frames from a camera RTSP stream and tracks objects."""

    def __init__(
        self,
        camera_id: int,
        camera_name: str,
        on_object_appear: Callable,
        on_object_depart: Callable,
        on_snapshot: Callable,
        target_classes: list[str] | None = None,
        detection_settings: dict | None = None,
        global_confidence: float = 0.5,
        zones: Optional[list[list[tuple[int, int]]]] = None,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._on_object_appear = on_object_appear
        self._on_object_depart = on_object_depart
        self._on_snapshot = on_snapshot
        self._target_classes = target_classes
        self._detection_settings = detection_settings or {}
        self._global_confidence = global_confidence
        self._zones = zones
        # Ignore zones: suppress detections of specific classes in defined regions
        self._ignore_zones = self._detection_settings.get("ignore_zones", [])
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tracks: dict[int, TrackedObject] = {}
        self._next_track_id = 1
        self._identity_memory: list[DepartedIdentity] = []
        # BoT-SORT: previous grayscale frame for GMC + optical flow
        self._prev_gray: Optional[np.ndarray] = None
        self._bootstrap_complete = False
        self._bootstrap_slot_held = False
        self._metrics = {
            "frames_decoded": 0,
            "motion_checks": 0,
            "motion_hits": 0,
            "yolo_runs": 0,
            "idle_skips": 0,
            "flow_updates": 0,
            "stationary_verifies": 0,
            "bootstrap_waits": 0,
            "bootstrap_runs": 0,
            "last_detection_mode": "idle",
            "last_detection_ts": 0.0,
            "start_time": time.monotonic(),
        }

    @property
    def is_tracking(self) -> bool:
        return self._running

    @property
    def active_tracks(self) -> dict[int, TrackedObject]:
        return self._tracks

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._tracking_loop())
        logger.info("Object tracking started for camera %s", self.camera_name)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Object tracking stopped for camera %s", self.camera_name)

    async def _tracking_loop(self):
        """Run object tracking in a thread to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        backoff = 5
        consecutive_failures = 0
        while self._running:
            try:
                await asyncio.to_thread(self._process_frames, loop)
                backoff = 5
                consecutive_failures = 0
            except Exception as e:
                consecutive_failures += 1
                is_conn_error = "NotFound" in type(e).__name__ or "timeout" in str(e).lower() or "Connection refused" in str(e)
                if is_conn_error and consecutive_failures >= 3:
                    # Camera is offline — back off quietly, max 5 min
                    backoff = min(backoff * 2, 300)
                    if consecutive_failures == 3:
                        logger.warning("Camera %s appears offline, backing off to %ds retries", self.camera_name, backoff)
                    elif consecutive_failures % 10 == 0:
                        logger.info("Camera %s still offline (%d attempts)", self.camera_name, consecutive_failures)
                else:
                    backoff = min(backoff * 2, 60)
                    logger.error(
                        "Object tracking error for camera %s: %s\n%s",
                        self.camera_name, e, traceback.format_exc(),
                    )
                await asyncio.sleep(backoff)

    def _create_zone_mask(self, width: int, height: int) -> Optional[np.ndarray]:
        """Create a binary mask from polygon zones."""
        if not self._zones:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        for zone in self._zones:
            pts = np.array(zone, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _bbox_in_zone(self, bbox: tuple, zone_mask: np.ndarray) -> bool:
        """Check if the center of a bbox falls within a zone mask."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        h, w = zone_mask.shape
        if 0 <= cy < h and 0 <= cx < w:
            return zone_mask[cy, cx] > 0
        return False

    def _is_in_ignore_zone(self, det: "Detection") -> bool:
        """Check if a detection falls within a per-camera ignore zone for its class."""
        if not self._ignore_zones:
            return False
        cx = (det.bbox[0] + det.bbox[2]) // 2
        cy = (det.bbox[1] + det.bbox[3]) // 2
        for zone in self._ignore_zones:
            classes = zone.get("classes", [])
            if classes and det.class_name not in classes:
                continue  # Zone doesn't apply to this class
            polygon = zone.get("polygon", [])
            if not polygon or len(polygon) < 3:
                continue
            pts = np.array(polygon, dtype=np.int32)
            if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
                return True
        return False

    def _detect_motion(self, prev_gray: np.ndarray, gray: np.ndarray) -> bool:
        """Cheap frame-differencing motion check on already-available grayscale.

        Returns True if motion exceeds MOTION_THRESHOLD (foreground pixel ratio).
        Much cheaper than YOLO — just absdiff + threshold.
        """
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        fg_ratio = np.count_nonzero(thresh) / thresh.size
        return fg_ratio > MOTION_THRESHOLD

    @staticmethod
    def _plan_detection_mode(
        *,
        is_first_frame: bool,
        bootstrap_complete: bool,
        startup_elapsed: float,
        startup_stagger: float,
        bootstrap_slot_held: bool,
        motion_active: bool,
        has_tracks: bool,
        time_since_detect: float,
    ) -> str:
        """Return the tracker mode for the current frame.

        Modes: bootstrap-wait, bootstrap-detect, motion-detect,
        stationary-verify, flow-only, idle.
        """
        if is_first_frame and not bootstrap_complete:
            if startup_elapsed < startup_stagger:
                return "bootstrap-wait"
            if bootstrap_slot_held:
                return "bootstrap-detect"
            return "bootstrap-wait"
        if motion_active and time_since_detect >= TRACK_INTERVAL_ACTIVE:
            return "motion-detect"
        if has_tracks and time_since_detect >= STATIONARY_VERIFY_INTERVAL:
            return "stationary-verify"
        if has_tracks:
            return "flow-only"
        return "idle"

    def get_metrics_snapshot(self) -> dict:
        uptime = max(time.monotonic() - self._metrics["start_time"], 0.001)
        return {
            **{k: v for k, v in self._metrics.items() if k != "start_time"},
            "active_tracks": len([t for t in self._tracks.values() if t.confirmed]),
            "bootstrap_complete": self._bootstrap_complete,
            "fps_decoded": round(self._metrics["frames_decoded"] / uptime, 2),
            "yolo_per_minute": round(self._metrics["yolo_runs"] * 60.0 / uptime, 2),
            "motion_hit_rate": round(
                self._metrics["motion_hits"] / max(self._metrics["motion_checks"], 1), 3
            ),
        }

    def _process_frames(self, loop):
        """Blocking frame processing loop — runs in a thread.

        Motion-gated detection with three modes:
          - Motion detected → YOLO at TRACK_INTERVAL_ACTIVE (1.5s)
          - Tracks exist, no motion → coast with optical flow, YOLO verify every 30s
          - No tracks, no motion → skip YOLO, only check motion every 0.3s
        """
        rtsp_url = stream_manager.get_rtsp_url(self.camera_name)
        logger.info("Object tracker connecting to RTSP: %s", rtsp_url)
        container = av.open(rtsp_url, options={"rtsp_transport": "tcp"})
        logger.info("Object tracker RTSP connected for %s", self.camera_name)

        zone_mask = None
        frame_count = 0
        start_time = time.monotonic()
        startup_stagger = min(2.0, (self.camera_id % 8) * 0.25)
        last_inference_time = 0.0
        last_flow_time = 0.0
        last_motion_check = 0.0
        motion_consecutive = 0
        motion_active = False
        last_motion_time = 0.0

        # Compute minimum confidence for initial pass
        per_obj_conf = {}
        if self._target_classes:
            for cls in self._target_classes:
                obj_s = self._detection_settings.get(cls, {})
                per_obj_conf[cls] = obj_s.get("confidence", self._global_confidence)
        min_confidence = min(per_obj_conf.values()) if per_obj_conf else self._global_confidence
        has_small_targets = bool(
            set(self._target_classes or []) & SMALL_OBJECT_CLASSES
        )
        last_enhanced_time = 0.0

        try:
            for frame in container.decode(video=0):
                if not self._running:
                    break

                frame_count += 1
                self._metrics["frames_decoded"] += 1
                now = time.monotonic()

                # ── Motion-gated scheduling ──
                # Check motion cheaply every MOTION_DETECT_INTERVAL
                need_motion_check = (
                    self._prev_gray is not None
                    and (now - last_motion_check) >= MOTION_DETECT_INTERVAL
                )
                # Need flow update for active tracks
                need_flow = (
                    self._tracks
                    and self._prev_gray is not None
                    and (now - last_flow_time) >= FLOW_INTERVAL
                )
                # First frame: must store gray for future motion checks
                is_first_frame = self._prev_gray is None

                if not need_motion_check and not need_flow and not is_first_frame:
                    self._metrics["idle_skips"] += 1
                    continue  # Skip — nothing to do

                # Convert frame (only when we actually need it)
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if zone_mask is None and self._zones:
                    h, w = img.shape[:2]
                    zone_mask = self._create_zone_mask(w, h)

                if frame_count == 1:
                    logger.info("Object tracker receiving frames for %s", self.camera_name)

                # ── Step 1: Motion check (very cheap) ──
                if need_motion_check:
                    self._metrics["motion_checks"] += 1
                    has_motion = self._detect_motion(self._prev_gray, gray)
                    last_motion_check = now
                    if has_motion:
                        self._metrics["motion_hits"] += 1
                        motion_consecutive += 1
                        if motion_consecutive >= MOTION_FRAMES_REQUIRED:
                            if not motion_active:
                                logger.debug("Motion detected on %s", self.camera_name)
                            motion_active = True
                            last_motion_time = now
                    else:
                        motion_consecutive = 0
                        # Cool down: motion_active stays True briefly after motion stops
                        if motion_active and (now - last_motion_time) > MOTION_COOLDOWN:
                            motion_active = False

                # ── Step 2: Decide if we need YOLO detection ──
                time_since_detect = now - last_inference_time
                if is_first_frame and not self._bootstrap_complete and not self._bootstrap_slot_held:
                    self._bootstrap_slot_held = BOOTSTRAP_SCHEDULER.try_acquire(self.camera_id)
                    if not self._bootstrap_slot_held:
                        self._metrics["bootstrap_waits"] += 1

                detection_mode = self._plan_detection_mode(
                    is_first_frame=is_first_frame,
                    bootstrap_complete=self._bootstrap_complete,
                    startup_elapsed=now - start_time,
                    startup_stagger=startup_stagger,
                    bootstrap_slot_held=self._bootstrap_slot_held,
                    motion_active=motion_active,
                    has_tracks=bool(self._tracks),
                    time_since_detect=time_since_detect,
                )
                self._metrics["last_detection_mode"] = detection_mode
                need_detection = detection_mode in {
                    "bootstrap-detect", "motion-detect", "stationary-verify"
                }

                if detection_mode == "stationary-verify":
                    self._metrics["stationary_verifies"] += 1
                    logger.debug(
                        "Stationary verify on %s (%d tracks)",
                        self.camera_name, len(self._tracks),
                    )

                if need_detection:
                    # ── BoT-SORT: Global Motion Compensation before association ──
                    gmc_transform = None
                    if self._prev_gray is not None:
                        gmc_transform = self._compute_gmc(self._prev_gray, gray)

                    # ── YOLO detection ──
                    yolo_classes = list(self._target_classes) if self._target_classes else None
                    if yolo_classes and 'cat' in yolo_classes and 'dog' not in yolo_classes:
                        yolo_classes.append('dog')

                    future = asyncio.run_coroutine_threadsafe(
                        object_detector.detect(
                            img,
                            confidence_threshold=min_confidence,
                            target_classes=yolo_classes,
                        ),
                        loop,
                    )
                    try:
                        detections = future.result(timeout=10)
                    except Exception:
                        detections = []

                    # Apply per-class confidence and min_area filters
                    filtered = []
                    for det in detections:
                        if (
                            det.class_name == 'dog'
                            and self._target_classes
                            and 'cat' in self._target_classes
                            and 'dog' not in self._target_classes
                        ):
                            det = Detection('cat', det.confidence * 0.95, det.bbox, det.crop)

                        obj_s = self._detection_settings.get(det.class_name, {})
                        required_conf = obj_s.get("confidence", self._global_confidence)
                        min_area = obj_s.get("min_area", 0)
                        if det.confidence < required_conf:
                            continue
                        bbox_area = (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1])
                        if bbox_area < min_area:
                            continue
                        if zone_mask is not None and not self._bbox_in_zone(det.bbox, zone_mask):
                            continue
                        if self._is_in_ignore_zone(det):
                            continue
                        filtered.append(det)

                    # Enhanced detection for small animals
                    # Use shorter interval when actively tracking small objects
                    enhanced_interval = ENHANCED_SCAN_INTERVAL
                    if self._tracks and any(
                        t.class_name in SMALL_OBJECT_CLASSES for t in self._tracks.values()
                    ):
                        enhanced_interval = ENHANCED_SCAN_ACTIVE
                    if has_small_targets and (now - last_enhanced_time) >= enhanced_interval:
                        small_classes = [c for c in (self._target_classes or []) if c in SMALL_OBJECT_CLASSES]
                        has_small_det = any(d.class_name in SMALL_OBJECT_CLASSES for d in filtered)
                        if not has_small_det and small_classes:
                            enhanced = self._enhanced_small_detect(img, small_classes, per_obj_conf, loop)
                            if enhanced:
                                logger.info(
                                    "Enhanced scan found %d small object(s) on %s",
                                    len(enhanced), self.camera_name,
                                )
                                filtered.extend(enhanced)
                        last_enhanced_time = now

                    # ── Full association (with GMC) ──
                    self._update_tracks(filtered, img, now, loop, gmc_transform)
                    last_inference_time = now
                    last_flow_time = now
                    self._metrics["yolo_runs"] += 1
                    self._metrics["last_detection_ts"] = now
                    if detection_mode == "bootstrap-detect":
                        self._metrics["bootstrap_runs"] += 1
                        self._bootstrap_complete = True
                        if self._bootstrap_slot_held:
                            BOOTSTRAP_SCHEDULER.release(self.camera_id)
                            self._bootstrap_slot_held = False

                elif need_flow and not need_detection:
                    # ── Inter-detection optical flow update (lightweight) ──
                    self._optical_flow_update(self._prev_gray, gray)
                    last_flow_time = now
                    self._metrics["flow_updates"] += 1

                # Store frame for next flow/GMC computation
                self._prev_gray = gray

        finally:
            if self._bootstrap_slot_held:
                BOOTSTRAP_SCHEDULER.release(self.camera_id)
                self._bootstrap_slot_held = False
            container.close()

    def _update_tracks(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        now: float,
        loop,
        gmc_transform=None,
    ):
        """BoT-SORT: two-stage association with Kalman + GMC + appearance.

        For person tracks, the cost matrix combines IoU and ReID cosine distance
        (weighted by APPEARANCE_ALPHA).  Other classes use pure IoU as before.

        1. Predict new positions via Kalman.
        2. Apply BoT-SORT GMC to compensate camera motion.
        3. Compute ReID embeddings for person detections.
        4. First association: high-confidence dets → all tracks (hybrid cost).
        5. Second association: low-confidence dets → unmatched tracks (IoU only).
        6. Create new tracks for unmatched high-confidence detections.
        7. Expire stale tracks (store appearance in identity memory).
        """
        # Lazy import to avoid circular dependency at module load time
        from services.recognition_service import recognition_service

        # Predict new positions for all tracks
        for track in self._tracks.values():
            track.kalman.predict()

        # ── BoT-SORT: apply GMC to predicted positions ──
        if gmc_transform is not None:
            self._apply_gmc(gmc_transform)

        # ── Compute ReID embeddings for person detections ──
        # Only compute if there are existing person tracks or departed identities to match
        det_reid: dict[int, np.ndarray] = {}  # det_index → embedding
        reid_available = recognition_service.reid_available
        if reid_available:
            has_person_targets = any(
                t.class_name in APPEARANCE_CLASSES for t in self._tracks.values()
            ) or any(
                m.class_name in APPEARANCE_CLASSES for m in self._identity_memory
            )
            if has_person_targets:
                for i, det in enumerate(detections):
                    if det.class_name in APPEARANCE_CLASSES and det.crop is not None and det.crop.size > 0:
                        emb = recognition_service.compute_reid_embedding_sync(det.crop)
                        if emb is not None:
                            det_reid[i] = emb

        # Split detections by confidence
        high_dets = [(i, d) for i, d in enumerate(detections) if d.confidence >= BYTETRACK_HIGH_THRESH]
        low_dets = [(i, d) for i, d in enumerate(detections) if d.confidence < BYTETRACK_HIGH_THRESH]

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # === First association: high-confidence detections vs all tracks ===
        track_ids = list(self._tracks.keys())
        if high_dets and track_ids:
            cost_matrix = np.zeros((len(high_dets), len(track_ids)), dtype=np.float32)
            for di, (det_idx, det) in enumerate(high_dets):
                for ti, tid in enumerate(track_ids):
                    track = self._tracks[tid]
                    if det.class_name != track.class_name:
                        cost_matrix[di, ti] = 1.0  # impossible match
                        continue
                    # IoU component
                    pred_bbox = track.kalman.predicted_bbox
                    iou = self._compute_iou(det.bbox, pred_bbox)
                    iou_cost = 1.0 - iou

                    # Appearance component (person tracks with embeddings)
                    if (
                        det.class_name in APPEARANCE_CLASSES
                        and det_idx in det_reid
                        and track.reid_embedding is not None
                    ):
                        cos_sim = float(np.dot(det_reid[det_idx], track.reid_embedding))
                        cos_sim = max(cos_sim, 0.0)
                        if cos_sim < APPEARANCE_GATE:
                            cost_matrix[di, ti] = 1.0  # appearance mismatch gate
                            continue
                        app_cost = 1.0 - cos_sim
                        cost_matrix[di, ti] = APPEARANCE_ALPHA * iou_cost + (1.0 - APPEARANCE_ALPHA) * app_cost
                    else:
                        cost_matrix[di, ti] = iou_cost

            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            for ri, ci in zip(row_indices, col_indices):
                if cost_matrix[ri, ci] > (1.0 - BYTETRACK_IOU_THRESH):
                    continue  # Combined cost too high
                det_idx, det = high_dets[ri]
                tid = track_ids[ci]
                track = self._tracks[tid]
                track.kalman.update(det.bbox)
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.crop = det.crop
                track.last_seen = now
                track.consecutive += 1
                matched_track_ids.add(tid)
                matched_det_indices.add(det_idx)

                # Update segmentation mask if available
                if hasattr(det, 'mask') and det.mask is not None:
                    track.mask = det.mask

                # EMA-smooth ReID embedding for the matched track
                if det_idx in det_reid:
                    new_emb = det_reid[det_idx]
                    if track.reid_embedding is not None:
                        track.reid_embedding = APPEARANCE_EMA * track.reid_embedding + (1.0 - APPEARANCE_EMA) * new_emb
                        norm = np.linalg.norm(track.reid_embedding)
                        if norm > 0:
                            track.reid_embedding = track.reid_embedding / norm
                    else:
                        track.reid_embedding = new_emb

                if not track.confirmed and track.consecutive >= TRACK_CONFIRM_FRAMES:
                    track.confirmed = True
                if track.confirmed and not track.appear_notified:
                    track.appear_notified = True
                    # First GIF frame
                    self._collect_gif_crop(track, det, now)
                    asyncio.run_coroutine_threadsafe(
                        self._on_object_appear(self.camera_id, self.camera_name, frame, det, track.track_id),
                        loop,
                    )
                # Collect GIF crops periodically
                if track.confirmed and (now - track.last_gif_crop_time) >= GIF_CROP_INTERVAL:
                    self._collect_gif_crop(track, det, now)
                # Retry recognition for unnamed objects (no new event created)
                if track.confirmed and track.named_object_name is None and (now - track.last_snapshot_time) >= SNAPSHOT_INTERVAL:
                    track.last_snapshot_time = now
                    asyncio.run_coroutine_threadsafe(
                        self._on_snapshot(self.camera_id, self.camera_name, frame, det, track.track_id),
                        loop,
                    )

        # === Second association: low-confidence detections vs unmatched tracks ===
        unmatched_track_ids = [tid for tid in track_ids if tid not in matched_track_ids]
        if low_dets and unmatched_track_ids:
            cost2 = np.zeros((len(low_dets), len(unmatched_track_ids)), dtype=np.float32)
            for di, (det_idx, det) in enumerate(low_dets):
                for ti, tid in enumerate(unmatched_track_ids):
                    track = self._tracks[tid]
                    if det.class_name != track.class_name:
                        cost2[di, ti] = 1.0
                        continue
                    pred_bbox = track.kalman.predicted_bbox
                    iou = self._compute_iou(det.bbox, pred_bbox)
                    cost2[di, ti] = 1.0 - iou

            row2, col2 = linear_sum_assignment(cost2)
            for ri, ci in zip(row2, col2):
                if cost2[ri, ci] > (1.0 - BYTETRACK_SECOND_IOU):
                    continue
                det_idx, det = low_dets[ri]
                tid = unmatched_track_ids[ci]
                track = self._tracks[tid]
                track.kalman.update(det.bbox)
                track.bbox = det.bbox
                track.confidence = det.confidence
                track.crop = det.crop
                track.last_seen = now
                track.consecutive += 1
                matched_track_ids.add(tid)
                matched_det_indices.add(det_idx)

        # === Create new tracks for unmatched high-confidence detections ===
        for det_idx, det in high_dets:
            if det_idx in matched_det_indices:
                continue
            track = TrackedObject(
                track_id=self._next_track_id,
                class_name=det.class_name,
                bbox=det.bbox,
                confidence=det.confidence,
                crop=det.crop,
                first_seen=now,
                last_seen=now,
                last_snapshot_time=now,
                reid_embedding=det_reid.get(det_idx),
                mask=getattr(det, 'mask', None),
            )
            inherited = self._try_inherit_identity(track, now)
            if inherited:
                if track.event_id:
                    logger.info(
                        "Track %d inherited identity '%s' + event %d from departed track (continuing same event)",
                        track.track_id, inherited, track.event_id,
                    )
                else:
                    logger.debug(
                        "Track %d inherited identity '%s' from departed track",
                        track.track_id, inherited,
                    )
            self._tracks[self._next_track_id] = track
            self._next_track_id += 1

        # === Expire lost tracks ===
        expired = [
            tid for tid, t in self._tracks.items()
            if (now - t.last_seen) > CLASS_LOST_TIMEOUT.get(t.class_name, TRACK_LOST_TIMEOUT)
        ]
        for tid in expired:
            track = self._tracks.pop(tid)
            if track.named_object_name:
                # Named track going to identity memory — DON'T fire depart yet.
                # For pets, compute CNN embedding for appearance-based re-ID
                cnn_emb = None
                if track.class_name in SMALL_OBJECT_CLASSES and track.crop is not None and track.crop.size > 0:
                    try:
                        cnn_emb = np.array(
                            recognition_service._compute_embedding_best(track.crop),
                            dtype=np.float64,
                        )
                    except Exception:
                        pass
                self._identity_memory.append(DepartedIdentity(
                    class_name=track.class_name,
                    bbox=track.bbox,
                    named_object_name=track.named_object_name,
                    departed_at=now,
                    reid_embedding=track.reid_embedding,
                    cnn_embedding=cnn_emb,
                    event_id=track.event_id,
                    gif_crops=track.gif_crops,
                    first_seen=track.first_seen,
                ))
                logger.debug(
                    "Named track %d (%s) going to identity memory, holding event %s open",
                    tid, track.named_object_name, track.event_id,
                )
            elif track.confirmed and track.appear_notified:
                # Unnamed track — fire depart immediately
                asyncio.run_coroutine_threadsafe(
                    self._on_object_depart(self.camera_id, self.camera_name, track),
                    loop,
                )

        # Prune stale identity memory — fire depart for expired remembered tracks
        surviving = []
        for m in self._identity_memory:
            # Pets get longer memory (cats disappear behind furniture, reappear)
            mem_timeout = PET_IDENTITY_MEMORY_TIMEOUT if m.class_name in SMALL_OBJECT_CLASSES else IDENTITY_MEMORY_TIMEOUT
            if (now - m.departed_at) < mem_timeout:
                surviving.append(m)
            elif m.event_id:
                # Memory expired without re-acquisition — fire depart now
                logger.info(
                    "Identity memory expired for '%s' (event %d), firing depart",
                    m.named_object_name, m.event_id,
                )
                # Build a minimal TrackedObject for the depart callback
                phantom = TrackedObject(
                    track_id=-1,
                    class_name=m.class_name,
                    bbox=m.bbox,
                    confidence=0.0,
                    crop=np.zeros((1, 1, 3), dtype=np.uint8),
                    first_seen=m.first_seen,
                    last_seen=m.departed_at,
                    named_object_name=m.named_object_name,
                    event_id=m.event_id,
                    gif_crops=m.gif_crops,
                )
                asyncio.run_coroutine_threadsafe(
                    self._on_object_depart(self.camera_id, self.camera_name, phantom),
                    loop,
                )
        self._identity_memory = surviving

    def _try_inherit_identity(
        self, track: TrackedObject, now: float
    ) -> str | None:
        """Check if a new track matches a recently-departed named identity.

        Uses appearance (ReID cosine similarity for persons, CNN embedding
        for pets) — REQUIRES appearance matching to prevent background objects
        from inheriting a departed person's identity.

        Pure IoU-only matching is disabled for persons (too risky for
        inanimate objects left behind after a person leaves).
        """
        best_score = 0.0
        best_name = None
        best_mem = None
        best_appearance_verified = False

        for mem in self._identity_memory:
            if mem.class_name != track.class_name:
                continue

            # Temporal constraint: if departed a long time ago, require stronger match
            age = now - mem.departed_at
            if age > 90 and track.confidence < 0.55:
                # Long-departed + low-confidence new detection = suspicious
                continue

            iou = self._compute_iou(track.bbox, mem.bbox)

            # Appearance-based matching (StrongSORT) for persons
            if (
                track.reid_embedding is not None
                and mem.reid_embedding is not None
            ):
                cos_sim = float(np.dot(track.reid_embedding, mem.reid_embedding))
                # Require stronger appearance match for older departures
                min_appearance = APPEARANCE_GATE
                if age > 60:
                    min_appearance = max(APPEARANCE_GATE, 0.30)
                score = 0.3 * iou + 0.7 * max(cos_sim, 0.0)
                if score > best_score and cos_sim > min_appearance:
                    best_score = score
                    best_name = mem.named_object_name
                    best_mem = mem
                    best_appearance_verified = True
            # CNN appearance matching for pets/animals
            elif (
                mem.cnn_embedding is not None
                and track.class_name in SMALL_OBJECT_CLASSES
                and track.crop is not None
                and track.crop.size > 0
            ):
                try:
                    track_emb = np.array(
                        recognition_service._compute_embedding_best(track.crop),
                        dtype=np.float64,
                    )
                    cos_sim = float(np.dot(track_emb, mem.cnn_embedding) / (
                        np.linalg.norm(track_emb) * np.linalg.norm(mem.cnn_embedding) + 1e-10
                    ))
                    # For pets, weight appearance higher since they look distinct
                    score = 0.2 * iou + 0.8 * max(cos_sim, 0.0)
                    if score > best_score and cos_sim > 0.45:
                        best_score = score
                        best_name = mem.named_object_name
                        best_mem = mem
                        best_appearance_verified = True
                except Exception:
                    pass  # CNN failed — do NOT fall through to IoU-only

            # REMOVED: IoU-only fallback for persons.
            # Background objects (chairs, decorations) at the same position as a
            # departed person would match on IoU alone, inheriting the person's
            # identity. This was the root cause of "objects in background recognised
            # as person".
            #
            # For non-person, non-pet classes (vehicles), allow IoU-only with strict threshold
            elif track.class_name not in ("person", "cat", "dog"):
                if iou > best_score and iou > 0.50:
                    best_score = iou
                    best_name = mem.named_object_name
                    best_mem = mem

        if best_name and best_mem:
            # For persons: REQUIRE appearance verification — never inherit on spatial overlap alone
            if track.class_name == "person" and not best_appearance_verified:
                logger.debug(
                    "Refusing to inherit identity '%s' for person track %d: no appearance verification",
                    best_name, track.track_id,
                )
                return None

            track.named_object_name = best_name
            # Inherit event continuity — reuse the same event instead of creating new
            if best_mem.event_id:
                track.event_id = best_mem.event_id
                track.appear_notified = True  # Skip on_object_appear — event exists
                track.confirmed = True
            if best_mem.gif_crops:
                track.gif_crops = best_mem.gif_crops
            if best_mem.first_seen:
                track.first_seen = best_mem.first_seen
            # Remove from memory so it's not matched again
            self._identity_memory = [m for m in self._identity_memory if m is not best_mem]
        return best_name

    def _compute_gmc(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        """BoT-SORT Global Motion Compensation: estimate camera ego-motion.

        Uses Shi-Tomasi corners + Lucas-Kanade flow + RANSAC affine estimation
        to compute a 2×3 affine transform representing camera motion between frames.
        Returns None if motion is negligible or insufficient features.
        """
        if prev_gray.shape != curr_gray.shape:
            return None
        pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=GMC_MAX_CORNERS,
            qualityLevel=GMC_QUALITY_LEVEL,
            minDistance=GMC_MIN_DISTANCE,
        )
        if pts is None or len(pts) < 10:
            return None

        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **LK_PARAMS,
        )
        good_old = pts[status.flatten() == 1]
        good_new = pts_new[status.flatten() == 1]

        if len(good_old) < 6:
            return None

        # Estimate partial affine (rotation + translation + uniform scale)
        transform, inliers = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if transform is None:
            return None

        # Check if motion is negligible (< 0.5px translation)
        tx, ty = transform[0, 2], transform[1, 2]
        if abs(tx) < 0.5 and abs(ty) < 0.5:
            return None

        return transform

    def _apply_gmc(self, transform: np.ndarray):
        """Apply camera motion compensation to all track Kalman states and bboxes.

        Adjusts both the Kalman filter center position and the stored bbox
        so that predictions align with the new camera viewpoint.
        """
        for track in self._tracks.values():
            # Transform Kalman center (cx, cy)
            kf = track.kalman.kf
            cx, cy = float(kf.statePost[0, 0]), float(kf.statePost[1, 0])
            new_pos = transform @ np.array([cx, cy, 1.0], dtype=np.float64)
            kf.statePost[0, 0] = float(new_pos[0])
            kf.statePost[1, 0] = float(new_pos[1])

            # Transform stored bbox corners
            x1, y1, x2, y2 = track.bbox
            corners = np.array([[x1, y1], [x2, y2]], dtype=np.float64)
            ones = np.ones((2, 1), dtype=np.float64)
            pts_h = np.hstack([corners, ones])
            new_corners = (transform @ pts_h.T).T
            track.bbox = (
                int(new_corners[0, 0]), int(new_corners[0, 1]),
                int(new_corners[1, 0]), int(new_corners[1, 1]),
            )

    def _optical_flow_update(self, prev_gray: np.ndarray, curr_gray: np.ndarray):
        """Lightweight inter-detection position update via Lucas-Kanade flow.

        For each active track, computes flow at the bbox corners + center
        and applies the average displacement to the Kalman state. This gives
        ~6× more frequent position updates between YOLO detections.
        Does NOT update last_seen — only YOLO detections refresh that.
        """
        if not self._tracks:
            return
        if prev_gray.shape != curr_gray.shape:
            return

        # Build sparse point set: 5 points per track (4 corners + center)
        points = []
        track_ids = []
        for tid, track in self._tracks.items():
            x1, y1, x2, y2 = track.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, cy)]:
                points.append([px, py])
                track_ids.append(tid)

        if not points:
            return

        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **LK_PARAMS,
        )

        # Group displacements by track
        displacements: dict[int, list[tuple[float, float]]] = {}
        for i, (tid, st) in enumerate(zip(track_ids, status.flatten())):
            if st == 1:
                dx = float(pts_new[i][0][0] - pts[i][0][0])
                dy = float(pts_new[i][0][1] - pts[i][0][1])
                if tid not in displacements:
                    displacements[tid] = []
                displacements[tid].append((dx, dy))

        # Apply median displacement to each track
        for tid, dxdy_list in displacements.items():
            if tid not in self._tracks or len(dxdy_list) < 3:
                continue
            avg_dx = float(np.median([d[0] for d in dxdy_list]))
            avg_dy = float(np.median([d[1] for d in dxdy_list]))

            # Skip tiny motions (sub-pixel noise)
            if abs(avg_dx) < 0.3 and abs(avg_dy) < 0.3:
                continue

            track = self._tracks[tid]
            x1, y1, x2, y2 = track.bbox
            track.bbox = (
                int(x1 + avg_dx), int(y1 + avg_dy),
                int(x2 + avg_dx), int(y2 + avg_dy),
            )
            # Update Kalman center directly (not a full measurement update)
            kf = track.kalman.kf
            kf.statePost[0, 0] += avg_dx
            kf.statePost[1, 0] += avg_dy

    def _collect_gif_crop(self, track: TrackedObject, det: Detection, now: float):
        """Collect a resized crop for GIF/timelapse generation."""
        crop = det.crop
        if crop is None or crop.size == 0:
            return
        h, w = crop.shape[:2]
        if w > 200:
            s = 200 / w
            crop = cv2.resize(crop, (200, int(h * s)))
        else:
            crop = crop.copy()
        if len(track.gif_crops) < MAX_GIF_CROPS:
            track.gif_crops.append(crop)
        track.last_gif_crop_time = now

    def _enhanced_small_detect(
        self,
        frame: np.ndarray,
        target_classes: list[str],
        per_obj_conf: dict[str, float],
        loop,
    ) -> list[Detection]:
        """Run YOLO on quadrant tiles for better small-object resolution.

        Splits the frame into 4 overlapping quadrants, giving small animals
        ~4× more pixels in the 640×640 YOLO input.  Also includes 'dog' in
        the scan and reclassifies dog→cat when appropriate.
        """
        h, w = frame.shape[:2]
        if w < 640 or h < 320:
            return []

        # Use a reduced confidence floor for the enhanced pass (50% of normal)
        min_conf = min(per_obj_conf.get(c, self._global_confidence) for c in target_classes) * 0.5

        # Include dog in scan classes so we can reclassify dog→cat
        scan_classes = list(target_classes)
        if 'cat' in scan_classes and 'dog' not in scan_classes:
            scan_classes.append('dog')

        # 4 overlapping quadrants
        pad = int(min(w, h) * 0.05)
        qw, qh = w // 2, h // 2
        tiles = [
            (0, 0, qw + pad, qh + pad),
            (qw - pad, 0, w, qh + pad),
            (0, qh - pad, qw + pad, h),
            (qw - pad, qh - pad, w, h),
        ]

        all_dets: list[Detection] = []
        for tx1, ty1, tx2, ty2 in tiles:
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(w, tx2), min(h, ty2)
            crop = frame[ty1:ty2, tx1:tx2]
            future = asyncio.run_coroutine_threadsafe(
                object_detector.detect(crop, confidence_threshold=min_conf, target_classes=scan_classes),
                loop,
            )
            try:
                tile_dets = future.result(timeout=10)
            except Exception:
                continue
            for d in tile_dets:
                cls = d.class_name
                conf = d.confidence
                # Reclassify dog→cat when camera has cat but not dog
                if cls == 'dog' and 'cat' in target_classes and 'dog' not in target_classes:
                    cls = 'cat'
                    conf *= 0.95

                obj_s = self._detection_settings.get(cls, {})
                req_conf = obj_s.get("confidence", self._global_confidence) * 0.6
                if conf < req_conf:
                    continue

                # Remap bbox from tile coords to full-frame
                bx1, by1, bx2, by2 = d.bbox
                fx1 = max(0, bx1 + tx1)
                fy1 = max(0, by1 + ty1)
                fx2 = min(w, bx2 + tx1)
                fy2 = min(h, by2 + ty1)
                if fx2 > fx1 and fy2 > fy1:
                    full_crop = frame[fy1:fy2, fx1:fx2].copy()
                    all_dets.append(Detection(cls, conf, (fx1, fy1, fx2, fy2), full_crop))

        # NMS to deduplicate detections in overlap regions
        if len(all_dets) > 1:
            boxes = np.array([d.bbox for d in all_dets], dtype=np.float32)
            scores = np.array([d.confidence for d in all_dets], dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, 0.45)
            if len(indices) > 0:
                all_dets = [all_dets[i] for i in indices.flatten()]
            else:
                all_dets = []

        return all_dets

    @staticmethod
    def _compute_iou(box1: tuple, box2: tuple) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0


class ObjectTracker:
    """Manages object tracking for all cameras."""

    def __init__(self):
        self._trackers: dict[int, CameraObjectTracker] = {}

    async def start_tracking(
        self,
        camera_id: int,
        camera_name: str,
        on_object_appear: Callable,
        on_object_depart: Callable,
        on_snapshot: Callable,
        target_classes: list[str] | None = None,
        detection_settings: dict | None = None,
        global_confidence: float = 0.5,
        zones: Optional[list] = None,
    ):
        if camera_id in self._trackers:
            await self.stop_tracking(camera_id)

        tracker = CameraObjectTracker(
            camera_id=camera_id,
            camera_name=camera_name,
            on_object_appear=on_object_appear,
            on_object_depart=on_object_depart,
            on_snapshot=on_snapshot,
            target_classes=target_classes,
            detection_settings=detection_settings,
            global_confidence=global_confidence,
            zones=zones,
        )
        self._trackers[camera_id] = tracker
        await tracker.start()

    async def stop_tracking(self, camera_id: int):
        tracker = self._trackers.pop(camera_id, None)
        if tracker:
            await tracker.stop()

    async def stop_all(self):
        for camera_id in list(self._trackers.keys()):
            await self.stop_tracking(camera_id)

    def is_tracking(self, camera_id: int) -> bool:
        t = self._trackers.get(camera_id)
        return t.is_tracking if t else False

    def get_active_tracks(self, camera_id: int) -> dict:
        t = self._trackers.get(camera_id)
        if not t:
            return {}
        return {
            tid: {
                "class_name": tr.class_name,
                "confidence": round(tr.confidence, 3),
                "bbox": tr.bbox,
                "first_seen": tr.first_seen,
                "last_seen": tr.last_seen,
                "confirmed": tr.confirmed,
                "named_object_name": tr.named_object_name,
            }
            for tid, tr in t.active_tracks.items()
            if tr.confirmed
        }

    def get_metrics_snapshot(self) -> dict:
        cameras = {
            cam_id: tracker.get_metrics_snapshot()
            for cam_id, tracker in self._trackers.items()
        }
        return {
            "bootstrap_scheduler": BOOTSTRAP_SCHEDULER.snapshot(),
            "active_trackers": len(self._trackers),
            "cameras": cameras,
            "totals": {
                "frames_decoded": sum(m["frames_decoded"] for m in cameras.values()),
                "yolo_runs": sum(m["yolo_runs"] for m in cameras.values()),
                "motion_checks": sum(m["motion_checks"] for m in cameras.values()),
                "motion_hits": sum(m["motion_hits"] for m in cameras.values()),
                "idle_skips": sum(m["idle_skips"] for m in cameras.values()),
            },
        }

    def is_named_object_recently_departed(self, camera_id: int, named_object_name: str) -> bool:
        """Check if a named object recently departed this camera (in identity memory).

        Used to suppress re-notifications when a track cycles (depart → reappear).
        """
        t = self._trackers.get(camera_id)
        if not t:
            return False
        now = time.monotonic()
        for mem in t._identity_memory:
            if (
                mem.named_object_name == named_object_name
                and (now - mem.departed_at) < IDENTITY_MEMORY_TIMEOUT
            ):
                return True
        return False

    def set_track_name(self, camera_id: int, class_name: str, bbox: tuple, name: str):
        """Set named_object_name on the track whose bbox best matches."""
        t = self._trackers.get(camera_id)
        if not t:
            return
        best_iou = 0.3
        best_track = None
        for tr in t.active_tracks.values():
            if tr.class_name != class_name:
                continue
            iou = CameraObjectTracker._compute_iou(bbox, tr.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = tr
        if best_track:
            best_track.named_object_name = name

    def set_track_event_id(self, camera_id: int, track_id: int, event_id: int):
        """Set DB event_id on a tracked object (called after appear event is stored)."""
        t = self._trackers.get(camera_id)
        if not t:
            return
        tr = t.active_tracks.get(track_id)
        if tr:
            tr.event_id = event_id


# Singleton instance
object_tracker = ObjectTracker()
