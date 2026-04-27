"""BanusNas — Motion Detector: OpenCV MOG2 background subtraction per camera."""

import asyncio
import logging
import time
import traceback
from typing import Callable, Optional

import av
import cv2
import numpy as np

from services.stream_manager import stream_manager

logger = logging.getLogger(__name__)

# ── Configurable at runtime via performance settings ──
_FRAME_SKIP = 10    # Process every Nth frame (higher = less CPU)
_BLUR_KERNEL = 7    # Gaussian blur kernel size (must be odd)


class CameraMotionDetector:
    """Pulls frames from a camera's RTSP sub-stream and detects motion via MOG2."""

    def __init__(
        self,
        camera_id: int,
        camera_name: str,
        on_motion: Callable,
        sensitivity: int = 16,
        min_area: int = 500,
        cooldown: float = 5.0,
        zones: Optional[list[list[tuple[int, int]]]] = None,
        ptz_mode: bool = False,
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self._on_motion = on_motion
        self._sensitivity = sensitivity
        self._min_area = min_area
        self._cooldown = cooldown
        self._zones = zones
        self._ptz_mode = ptz_mode
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_motion_time = 0.0
        self._motion_active = False

    @property
    def is_detecting(self) -> bool:
        return self._running

    @property
    def motion_active(self) -> bool:
        return self._motion_active

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._detection_loop())
        logger.info("Motion detection started for camera %s", self.camera_name)

    async def _detection_loop(self):
        """Run MOG2 motion detection in a thread to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        backoff = 5
        while self._running:
            try:
                await asyncio.to_thread(self._process_frames, loop)
                backoff = 5
            except Exception as e:
                logger.error(
                    "Motion detection error for camera %s: %s\n%s",
                    self.camera_name, e, traceback.format_exc(),
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _create_zone_mask(self, width: int, height: int) -> Optional[np.ndarray]:
        """Create a binary mask from polygon zones."""
        if not self._zones:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        for zone in self._zones:
            pts = np.array(zone, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _process_frames(self, loop):
        """Blocking frame processing loop — runs in a thread."""
        rtsp_url = stream_manager.get_rtsp_url(self.camera_name)
        logger.info("Motion detector connecting to RTSP: %s", rtsp_url)
        container = av.open(rtsp_url, options={"rtsp_transport": "tcp"})
        logger.info("Motion detector RTSP connected for %s", self.camera_name)

        if self._ptz_mode:
            self._process_frames_ptz(container, loop)
        else:
            self._process_frames_standard(container, loop)

    def _process_frames_standard(self, container, loop):
        """Standard MOG2 motion detection for fixed cameras."""
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=self._sensitivity,
            history=300,
        )
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        zone_mask = None
        frame_count = 0
        motion_count = 0
        # Require 2 consecutive motion frames to trigger (debounce flicker)
        consecutive_motion = 0
        CONSECUTIVE_REQUIRED = 2

        try:
            for frame in container.decode(video=0):
                if not self._running:
                    break

                frame_count += 1
                if frame_count == 1:
                    logger.info("Motion detector receiving frames for %s", self.camera_name)

                # Process every Nth frame (~3 FPS from 30 FPS stream) to reduce CPU load
                if frame_count % _FRAME_SKIP != 0:
                    continue

                img = frame.to_ndarray(format="bgr24")
                # Downscale for faster processing
                small = cv2.resize(img, (640, 360))
                blurred = cv2.GaussianBlur(small, (_BLUR_KERNEL, _BLUR_KERNEL), 0)

                if zone_mask is None and self._zones:
                    zone_mask = self._create_zone_mask(640, 360)

                fg_mask = bg_subtractor.apply(blurred, learningRate=0.002)

                # Morphological operations: erode to remove small noise, then dilate
                fg_mask = cv2.erode(fg_mask, kernel_open, iterations=1)
                fg_mask = cv2.dilate(fg_mask, kernel_close, iterations=2)

                # Apply zone mask if configured
                if zone_mask is not None:
                    fg_mask = cv2.bitwise_and(fg_mask, zone_mask)

                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant = [c for c in contours if cv2.contourArea(c) > self._min_area]

                # Check that total foreground area is meaningful (>0.5% of frame)
                total_fg = sum(cv2.contourArea(c) for c in significant)
                frame_area = 640 * 360
                fg_ratio = total_fg / frame_area

                # Reject if foreground ratio is too high (global illumination change)
                if fg_ratio > 0.40:
                    consecutive_motion = 0
                    continue

                if len(significant) > 0 and fg_ratio > 0.005:
                    consecutive_motion += 1
                else:
                    consecutive_motion = 0

                motion_detected = consecutive_motion >= CONSECUTIVE_REQUIRED

                now = time.monotonic()
                if motion_detected and (now - self._last_motion_time) > self._cooldown:
                    self._last_motion_time = now
                    self._motion_active = True
                    motion_count += 1
                    # Compute motion metrics
                    motion_area = sum(cv2.contourArea(c) for c in significant)
                    motion_score = round(motion_area / (640 * 360), 4)
                    full_frame = frame.to_ndarray(format="bgr24")
                    fh, fw = full_frame.shape[:2]
                    sx, sy = fw / 640, fh / 360
                    motion_regions = [
                        (int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                        for x, y, w, h in (cv2.boundingRect(c) for c in significant)
                    ]
                    logger.info(
                        "Motion detected on %s (#%d, frame %d, score=%.4f, regions=%d)",
                        self.camera_name, motion_count, frame_count, motion_score, len(motion_regions),
                    )
                    asyncio.run_coroutine_threadsafe(
                        self._on_motion(self.camera_id, self.camera_name, full_frame, motion_score, motion_regions),
                        loop,
                    )
                elif not motion_detected and self._motion_active:
                    if (now - self._last_motion_time) > self._cooldown * 2:
                        self._motion_active = False

        finally:
            container.close()

    def _process_frames_ptz(self, container, loop):
        """PTZ-aware motion detection: ignores camera movement, only triggers on objects.

        Uses object tracking to suppress re-detections of static objects.
        Only triggers when a NEW object appears or an existing one has MOVED
        significantly since the last detection cycle.
        """
        from services.object_detector import object_detector

        frame_count = 0
        prev_gray = None
        detection_count = 0
        # Lower sample rate for PTZ — every 20th frame (~1.5 FPS from 30 FPS)
        sample_rate = 20
        # Track previously seen objects to suppress static re-detections
        # Each entry: {"class": str, "bbox": (x1,y1,x2,y2), "last_seen": float}
        prev_detections: list[dict] = []
        STATIC_IOU_THRESHOLD = 0.45  # Above this IoU = same static object
        SCENE_STABLE_TIMEOUT = 60.0  # Forget tracked objects after 60s of not seeing them

        try:
            for frame in container.decode(video=0):
                if not self._running:
                    break

                frame_count += 1
                if frame_count == 1:
                    logger.info("PTZ motion detector receiving frames for %s", self.camera_name)

                if frame_count % sample_rate != 0:
                    continue

                img = frame.to_ndarray(format="bgr24")
                small = cv2.resize(img, (640, 360))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                # Global motion check: if the camera is panning/tilting, the entire
                # frame changes uniformly.  Measure the fraction of changed pixels —
                # if > 60% it is camera movement, not a real object.
                # If < 0.5% nothing is moving — skip expensive YOLO inference.
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    changed_ratio = np.count_nonzero(thresh) / thresh.size

                    if changed_ratio > 0.60:
                        # Camera is moving — reset tracked objects since the scene changed
                        prev_gray = gray
                        prev_detections.clear()
                        continue

                    if changed_ratio < 0.005:
                        # Scene is completely static — no need for object detection
                        prev_gray = gray
                        continue

                prev_gray = gray

                now = time.monotonic()
                if (now - self._last_motion_time) < self._cooldown:
                    continue

                # Run object detection on the downscaled frame (640x360) to save CPU.
                # The detector will resize to 640x640 internally anyway.
                future = asyncio.run_coroutine_threadsafe(
                    object_detector.detect(img, confidence_threshold=0.3),
                    loop,
                )
                try:
                    detections = future.result(timeout=10)
                except Exception:
                    detections = []

                if not detections:
                    if self._motion_active and (now - self._last_motion_time) > self._cooldown * 2:
                        self._motion_active = False
                    continue

                # --- Filter out static objects that haven't moved ---
                # Expire old tracked objects
                prev_detections = [
                    p for p in prev_detections
                    if (now - p["last_seen"]) < SCENE_STABLE_TIMEOUT
                ]

                new_detections = []
                updated_tracked = list(prev_detections)

                for det in detections:
                    bbox = det.bbox
                    matched_static = False

                    for tracked in updated_tracked:
                        if tracked["class"] != det.class_name:
                            continue
                        iou = self._compute_iou(bbox, tracked["bbox"])
                        if iou > STATIC_IOU_THRESHOLD:
                            # Same object in same position — update its position and skip
                            tracked["bbox"] = bbox
                            tracked["last_seen"] = now
                            matched_static = True
                            break

                    if not matched_static:
                        # NEW object — hasn't been seen here before
                        new_detections.append(det)
                        updated_tracked.append({
                            "class": det.class_name,
                            "bbox": bbox,
                            "last_seen": now,
                        })

                prev_detections = updated_tracked

                if new_detections:
                    self._last_motion_time = now
                    self._motion_active = True
                    detection_count += 1

                    motion_regions = [
                        (d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1])
                        for d in new_detections
                    ]
                    fh, fw = img.shape[:2]
                    motion_score = sum(
                        (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                        for d in new_detections
                    ) / max(fh * fw, 1)

                    logger.info(
                        "PTZ new object on %s (#%d, frame %d, new=%d, static=%d)",
                        self.camera_name, detection_count, frame_count,
                        len(new_detections), len(detections) - len(new_detections),
                    )
                    asyncio.run_coroutine_threadsafe(
                        self._on_motion(self.camera_id, self.camera_name, img, motion_score, motion_regions),
                        loop,
                    )
                elif self._motion_active:
                    if (now - self._last_motion_time) > self._cooldown * 2:
                        self._motion_active = False

        finally:
            container.close()

    @staticmethod
    def _compute_iou(box1: tuple, box2: tuple) -> float:
        """Compute IoU between two (x1,y1,x2,y2) bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / max(union, 1e-6)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Motion detection stopped for camera %s", self.camera_name)

    def update_zones(self, zones: Optional[list[list[tuple[int, int]]]]):
        self._zones = zones

    def update_sensitivity(self, sensitivity: int, min_area: int):
        self._sensitivity = sensitivity
        self._min_area = min_area


class MotionDetector:
    """Manages motion detection for all cameras."""

    def __init__(self):
        self._detectors: dict[int, CameraMotionDetector] = {}

    async def start_detection(
        self,
        camera_id: int,
        camera_name: str,
        on_motion: Callable,
        zones: Optional[list] = None,
        ptz_mode: bool = False,
    ):
        if camera_id in self._detectors:
            await self.stop_detection(camera_id)

        detector = CameraMotionDetector(
            camera_id=camera_id,
            camera_name=camera_name,
            on_motion=on_motion,
            zones=zones,
            ptz_mode=ptz_mode,
        )
        self._detectors[camera_id] = detector
        await detector.start()

    async def stop_detection(self, camera_id: int):
        detector = self._detectors.pop(camera_id, None)
        if detector:
            await detector.stop()

    async def stop_all(self):
        for camera_id in list(self._detectors.keys()):
            await self.stop_detection(camera_id)

    def is_detecting(self, camera_id: int) -> bool:
        d = self._detectors.get(camera_id)
        return d.is_detecting if d else False

    def motion_active(self, camera_id: int) -> bool:
        d = self._detectors.get(camera_id)
        return d.motion_active if d else False


# Singleton instance
motion_detector = MotionDetector()
