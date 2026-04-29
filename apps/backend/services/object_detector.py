"""BanusNas — Object Detector: YOLO26n via OpenVINO (Intel iGPU / CPU)."""

import asyncio
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from core.config import settings

logger = logging.getLogger(__name__)

# COCO class names used by YOLOv8
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "package",
]


class Detection:
    """Single object detection result, optionally with segmentation mask."""

    def __init__(self, class_name: str, confidence: float, bbox: tuple[int, int, int, int], crop: np.ndarray, mask: np.ndarray = None):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.crop = crop
        self.mask = mask  # Optional: bbox-cropped binary mask (H, W), uint8 0/255

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": {"x1": self.bbox[0], "y1": self.bbox[1], "x2": self.bbox[2], "y2": self.bbox[3]},
        }


class ObjectDetector:
    """YOLO26n object detection using OpenVINO (Intel iGPU with CPU fallback)."""

    def __init__(self):
        self._compiled_model = None
        self._input_layer = None
        self._output_layer = None
        self._proto_layer = None    # Seg model: prototype masks output
        self._input_shape: Optional[tuple] = None
        self._lock = asyncio.Lock()
        # Allow 2 concurrent inferences for better GPU pipelining (one executing, one preparing)
        self._semaphore = asyncio.Semaphore(2)
        # Coral TPU is single-threaded — serialize access to avoid "Infer Request is busy"
        self._coral_lock = asyncio.Lock()
        self._initialized = False
        self._device = "CPU"
        self._is_seg = False  # Auto-detected: True if model outputs masks
        # Preallocated letterbox canvas (reused across frames to reduce GC pressure)
        self._canvas: Optional[np.ndarray] = None
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "total_calls": 0,
            "openvino_calls": 0,
            "tiled_calls": 0,
            "coral_calls": 0,
            "remote_calls": 0,
            "coral_fallback_errors": 0,
            "coral_animal_fallbacks": 0,
            "remote_fallback_errors": 0,
            "total_detections": 0,
            "last_path": None,
            "last_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
        }

    def _record_metrics(self, *, path: str, latency_ms: float, detections: int, count_as_call: bool = True, **increments):
        with self._metrics_lock:
            if count_as_call:
                self._metrics["total_calls"] += 1
            self._metrics["total_detections"] += detections
            self._metrics["last_path"] = path
            self._metrics["last_latency_ms"] = round(latency_ms, 2)
            call_count = max(self._metrics["total_calls"], 1)
            prev_avg = self._metrics["avg_latency_ms"]
            self._metrics["avg_latency_ms"] = round(((prev_avg * (call_count - 1)) + latency_ms) / call_count, 2)
            for key, delta in increments.items():
                self._metrics[key] = self._metrics.get(key, 0) + delta

    def get_metrics_snapshot(self) -> dict:
        with self._metrics_lock:
            return dict(self._metrics)

    async def start(self):
        """Load the ONNX model. Gracefully skips if model file is missing."""
        try:
            await asyncio.to_thread(self._load_model)
        except Exception as e:
            logger.warning("Object detector disabled — model not available: %s", e)

    def _load_model(self):
        import os
        import openvino as ov

        model_path = settings.detector_model_path

        # Prefer seg model if one exists in the same directory
        import glob
        models_dir = os.path.dirname(model_path)
        seg_candidates = sorted(glob.glob(os.path.join(models_dir, "*-seg.onnx")))
        if seg_candidates:
            model_path = seg_candidates[0]
            logger.info("Segmentation model found, using: %s", model_path)

        # Limit threads to avoid saturating the N100's 4 cores
        os.environ.setdefault("OMP_NUM_THREADS", "2")

        core = ov.Core()
        available_devices = core.available_devices
        logger.info("OpenVINO available devices: %s", available_devices)

        # Read the ONNX model
        model = core.read_model(model_path)

        # Try to reshape input to configured resolution (default 1280 for 2K/4K cameras).
        # Seg models with hardcoded Reshape ops may not support this — fall back gracefully.
        input_size = settings.detector_input_size
        native_shape = model.input(0).shape
        if native_shape[2] != input_size or native_shape[3] != input_size:
            try:
                model.reshape({model.input(0): [1, 3, input_size, input_size]})
                logger.info("Reshaped YOLO input from %s to [1, 3, %d, %d]", native_shape, input_size, input_size)
            except Exception:
                logger.info("Model reshape to %d failed (seg model), using native %s — tiled detection will provide high-res coverage", input_size, native_shape)

        # Try GPU first, fall back to CPU
        device = "GPU" if "GPU" in available_devices else "CPU"
        try:
            # Set performance hint for throughput on GPU, latency on CPU
            config = {}
            if device == "GPU":
                config["PERFORMANCE_HINT"] = "THROUGHPUT"
            else:
                config["PERFORMANCE_HINT"] = "LATENCY"
                config["INFERENCE_NUM_THREADS"] = "2"

            self._compiled_model = core.compile_model(model, device, config)
            self._device = device
        except Exception as e:
            logger.warning("GPU compilation failed, falling back to CPU: %s", e)
            self._compiled_model = core.compile_model(
                model, "CPU", {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "2"}
            )
            self._device = "CPU"

        self._input_layer = self._compiled_model.input(0)
        self._output_layer = self._compiled_model.output(0)
        self._input_shape = self._input_layer.shape  # [1, 3, input_size, input_size]

        # Auto-detect segmentation model: seg models have 2+ outputs (detections + protos)
        num_outputs = len(self._compiled_model.outputs)
        if num_outputs > 1:
            self._proto_layer = self._compiled_model.output(1)
            self._is_seg = True
            logger.info("Segmentation model detected (%d outputs) — masks enabled", num_outputs)
        else:
            self._is_seg = False

        self._initialized = True
        logger.info("Object detector loaded: %s (device: %s, seg: %s)", model_path, self._device, self._is_seg)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize and normalize frame for YOLO input at configured resolution."""
        h, w = frame.shape[:2]
        target_size = self._input_shape[2]  # from settings.detector_input_size

        # Letterbox resize maintaining aspect ratio
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Reuse preallocated canvas to reduce GC pressure
        if self._canvas is None or self._canvas.shape[0] != target_size:
            self._canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        else:
            self._canvas[:] = 114
        pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
        self._canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Normalize to [0, 1] and transpose to CHW — use float32 division in-place
        blob = self._canvas.astype(np.float32)
        blob *= (1.0 / 255.0)
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        return blob, scale, pad_x, pad_y

    def _postprocess(
        self,
        outputs: np.ndarray,
        frame: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        confidence_threshold: float,
        target_classes: Optional[list[str]] = None,
        protos: Optional[np.ndarray] = None,
    ) -> list[Detection]:
        """Parse YOLO detection/seg output in either format.

        End-to-end (NMS-free) format:
          - (1, 300, 6): [x1, y1, x2, y2, conf, class_id]
          - (1, 300, 38): same + 32 mask coefficients (seg)

        Standard (multi-candidate) format:
          - (1, 116, 8400): [cx, cy, w, h, 80×class_probs, 32×mask_coeffs]
          - Requires transpose + NMS

        Proto shape (seg only): (1, 32, 160, 160).
        All coordinates are in the 640×640 letterboxed input space.
        """
        h, w = frame.shape[:2]
        predictions = outputs[0]
        if predictions.ndim == 3:
            predictions = predictions[0]

        # Prepare proto masks for seg model
        proto_data = None
        if self._is_seg and protos is not None:
            proto_data = protos[0] if protos.ndim == 4 else protos  # (32, ph, pw)

        # ── Detect output format and normalize to (N, 6+) corner format ──
        # Use the minimum per-class threshold as the NMS pre-filter so that
        # lower-confidence animal detections survive to be classified.
        nms_confidence = min(
            confidence_threshold,
            settings.detector_animal_min_confidence,
        )

        if predictions.shape[0] < predictions.shape[1]:
            # Standard format: (features, candidates) e.g. (116, 8400)
            predictions = self._nms_standard(predictions, nms_confidence, settings.detector_nms_iou_threshold)

        detections = []
        for pred in predictions:
            confidence = float(pred[4])
            class_id = int(pred[5])
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"

            # Per-class confidence thresholds
            if class_name == "person":
                min_conf = settings.detector_person_min_confidence
            elif class_name in ("cat", "dog"):
                min_conf = settings.detector_animal_min_confidence
            else:
                min_conf = confidence_threshold

            if confidence < min_conf:
                continue

            if target_classes and class_name not in target_classes:
                continue

            # Convert from padded/scaled coords back to original frame coords
            x1 = int((pred[0] - pad_x) / scale)
            y1 = int((pred[1] - pad_y) / scale)
            x2 = int((pred[2] - pad_x) / scale)
            y2 = int((pred[3] - pad_y) / scale)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Reject person detections that are too small — real people in
            # CCTV occupy a meaningful portion of the frame.  A tiny box is
            # almost always a mis-classified object fragment (remote, phone,
            # furniture edge).  Min area = 0.15% of frame (lowered from 0.5%
            # to avoid filtering out small children and distant subjects).
            box_area = (x2 - x1) * (y2 - y1)
            frame_area = h * w
            if class_name == "person" and box_area < frame_area * 0.0015:
                continue

            # Generate instance mask from seg model
            det_mask = None
            if proto_data is not None and pred.shape[0] > 6:
                mask_coeffs = pred[6:38].astype(np.float32)
                num_protos, ph, pw = proto_data.shape
                raw = mask_coeffs @ proto_data.reshape(num_protos, -1)
                raw = 1.0 / (1.0 + np.exp(-raw))  # sigmoid
                mask_160 = raw.reshape(ph, pw)
                mask_full = cv2.resize(mask_160, (w, h), interpolation=cv2.INTER_LINEAR)
                det_mask = (mask_full[y1:y2, x1:x2] > 0.5).astype(np.uint8)

            # Pad person crops for full body + head visibility (improves face detection)
            if class_name == "person":
                box_h = y2 - y1
                box_w = x2 - x1
                cy1 = max(0, y1 - int(box_h * 0.25))
                cy2 = min(h, y2 + int(box_h * 0.10))
                cx1 = max(0, x1 - int(box_w * 0.10))
                cx2 = min(w, x2 + int(box_w * 0.10))
                crop = frame[cy1:cy2, cx1:cx2].copy()
            else:
                crop = frame[y1:y2, x1:x2].copy()
                # Apply seg mask to non-person crops: zero out background
                # pixels so CNN embeddings focus on the actual object, not
                # surrounding context (reduces bag→cat type confusion).
                if det_mask is not None and det_mask.shape == crop.shape[:2]:
                    crop[det_mask == 0] = 0
            detections.append(Detection(class_name, confidence, (x1, y1, x2, y2), crop, det_mask))

        # Merge partial person fragments into full-body boxes
        detections = self._merge_contained_persons(detections)

        # Fix common YOLO misclassifications (e.g. cat detected as bird)
        detections = self._fix_confused_classes(detections, frame)

        return detections

    @staticmethod
    def _merge_contained_persons(detections: list["Detection"]) -> list["Detection"]:
        """When a smaller person box is mostly inside a larger person box,
        keep only the larger one.  YOLO sometimes produces a full-body box at
        moderate confidence AND a higher-confidence partial box (torso, hands).
        The tracker would pick the partial — we want the full-body box instead.
        """
        persons = [(i, d) for i, d in enumerate(detections) if d.class_name == "person"]
        if len(persons) < 2:
            return detections

        drop = set()
        for i, (idx_a, a) in enumerate(persons):
            ax1, ay1, ax2, ay2 = a.bbox
            a_area = (ax2 - ax1) * (ay2 - ay1)
            for j, (idx_b, b) in enumerate(persons):
                if i >= j:
                    continue
                bx1, by1, bx2, by2 = b.bbox
                b_area = (bx2 - bx1) * (by2 - by1)

                # Intersection
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                if ix1 >= ix2 or iy1 >= iy2:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)

                smaller_area = min(a_area, b_area)
                # If >60% of the smaller box is inside the larger one,
                # drop the smaller box (even if it has higher confidence).
                if inter > 0.6 * smaller_area:
                    if a_area < b_area:
                        drop.add(idx_a)
                    else:
                        drop.add(idx_b)

        if not drop:
            return detections
        return [d for i, d in enumerate(detections) if i not in drop]

    @staticmethod
    def _nms_standard(predictions: np.ndarray, confidence_threshold: float, iou_threshold: float = 0.7) -> np.ndarray:
        """Convert standard YOLO output (features, candidates) to NMS'd corner format.

        Input: (116, 8400) = [cx, cy, w, h, 80×class_probs, 32×mask_coeffs] per candidate.
        Output: (N, 38) = [x1, y1, x2, y2, conf, class_id, 32×mask_coeffs] after NMS.

        Uses per-class NMS so that nearby objects of the same class (e.g. two
        cats on a couch) are preserved when their IoU is below the threshold.
        """
        preds = predictions.T  # (8400, 116)
        num_classes = 80
        mask_start = 4 + num_classes  # 84

        # Best class per candidate
        class_probs = preds[:, 4:mask_start]
        max_conf = class_probs.max(axis=1)
        class_ids = class_probs.argmax(axis=1)

        # Filter by confidence
        keep = max_conf >= confidence_threshold
        preds = preds[keep]
        max_conf = max_conf[keep]
        class_ids = class_ids[keep]

        n_candidates = len(preds)
        if n_candidates == 0:
            return np.empty((0, 6), dtype=np.float32)

        # Convert cx, cy, w, h → x1, y1, x2, y2
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Per-class NMS — apply NMS within each class independently
        unique_classes = np.unique(class_ids)
        all_indices = []
        for cls_id in unique_classes:
            cls_indices = np.where(class_ids == cls_id)[0]
            cls_boxes = np.stack([x1[cls_indices], y1[cls_indices], bw[cls_indices], bh[cls_indices]], axis=1)
            cls_scores = max_conf[cls_indices]
            nms_idx = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(), cls_scores.tolist(),
                confidence_threshold, iou_threshold,
            )
            if len(nms_idx) > 0:
                all_indices.extend(cls_indices[nms_idx.flatten()].tolist())

        if not all_indices:
            return np.empty((0, 6), dtype=np.float32)

        # Build normalized output: [x1, y1, x2, y2, conf, class_id, mask_coeffs...]
        has_masks = preds.shape[1] > mask_start
        rows = []
        for i in all_indices:
            row = [x1[i], y1[i], x2[i], y2[i], max_conf[i], float(class_ids[i])]
            if has_masks:
                row.extend(preds[i, mask_start:].tolist())
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _fix_confused_classes(detections: list["Detection"], frame: np.ndarray) -> list["Detection"]:
        """Correct common COCO class confusions based on heuristics.

        Known issues:
        - Cats/dogs on the ground are sometimes classified as 'bird' by YOLO,
          especially at low resolution or unusual angles.
        - 'teddy bear' is often a sitting cat from above.
        - Small vehicles can be confused with animals.

        Heuristic: birds in CCTV are typically small and in the upper portion of
        the frame.  A 'bird' that is large and in the lower half is likely a
        misclassified cat/dog — reclassify it.  Also require higher confidence
        for bird detections.
        """
        frame_h, frame_w = frame.shape[:2]
        corrected = []

        for det in detections:
            # ── Drop COCO classes that are false-positive magnets in CCTV ──
            # Backpacks, handbags, suitcases are frequently confused with
            # animals (especially cats) by YOLO at common CCTV angles.
            if det.class_name in ("backpack", "handbag", "suitcase"):
                logger.debug("Dropping %s detection (false-positive-prone class)", det.class_name)
                continue

            if det.class_name == "bird":
                bw = det.bbox[2] - det.bbox[0]
                bh = det.bbox[3] - det.bbox[1]
                bbox_area = bw * bh
                frame_area = frame_h * frame_w
                area_ratio = bbox_area / max(frame_area, 1)
                center_y = (det.bbox[1] + det.bbox[3]) / 2

                # If the "bird" is large (>2% of frame), in the lower 60% of the
                # image, or has a wide aspect ratio, it's likely a cat/dog.
                is_large = area_ratio > 0.02
                is_low = center_y > frame_h * 0.40
                is_wide = bw > bh * 1.3  # Cats/dogs are wider than tall from side view

                if (is_large and is_low) or (is_low and is_wide):
                    logger.debug(
                        "Reclassifying bird -> cat (area=%.3f, y=%.0f/%.0f, aspect=%.1f)",
                        area_ratio, center_y, frame_h, bw / max(bh, 1),
                    )
                    det = Detection("cat", det.confidence * 0.95, det.bbox, det.crop)
                elif det.confidence < 0.55:
                    # Low-confidence bird detections are often false positives — skip
                    continue

            # 'teddy bear' misdetections — only reclassify to cat with stricter criteria:
            # must be low in frame, roughly square (compact), AND confidence ≥ 0.55
            elif det.class_name == "teddy bear":
                bw = det.bbox[2] - det.bbox[0]
                bh = det.bbox[3] - det.bbox[1]
                center_y = (det.bbox[1] + det.bbox[3]) / 2
                is_low = center_y > frame_h * 0.35
                is_compact = 0.6 < (bw / max(bh, 1)) < 1.7  # Roughly square = curled cat
                if is_low and is_compact and det.confidence >= 0.55:
                    logger.debug(
                        "Reclassifying teddy bear -> cat (compact, low, conf=%.2f)",
                        det.confidence,
                    )
                    det = Detection("cat", det.confidence * 0.85, det.bbox, det.crop)
                else:
                    continue  # Likely a false positive in CCTV

            corrected.append(det)

        return corrected

    async def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        target_classes: Optional[list[str]] = None,
        tiled: bool = False,
    ) -> list[Detection]:
        """Run object detection on a frame. Limits concurrency to avoid CPU saturation.

        When tiled=True, splits the frame into 4 overlapping quadrants for ~2x
        effective resolution.  The Coral path always uses tiled YOLO for the
        animal fallback internally.
        """
        if confidence_threshold is None:
            confidence_threshold = settings.detector_confidence_threshold
        start = time.perf_counter()

        # Remote ML offload
        from services.ml_client import ml_offload_enabled
        if ml_offload_enabled:
            detections = await self._detect_remote(frame, confidence_threshold, target_classes)
            self._record_metrics(
                path="remote",
                latency_ms=(time.perf_counter() - start) * 1000.0,
                detections=len(detections),
                remote_calls=1,
            )
            return detections

        # Coral Edge TPU (if enabled and available)
        if settings.coral_enabled:
            from services.coral_backend import coral_backend
            if coral_backend.available:
                coral_threshold = settings.coral_confidence_threshold
                detections = await self._detect_coral(frame, coral_threshold, target_classes)
                self._record_metrics(
                    path="coral",
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    detections=len(detections),
                    coral_calls=1,
                )
                return detections

        if not self._initialized:
            return []

        async with self._semaphore:
            if tiled and frame.shape[1] >= 640 and frame.shape[0] >= 320:
                detections = await asyncio.to_thread(
                    self._detect_tiled, frame, confidence_threshold, target_classes
                )
                self._record_metrics(
                    path="openvino-tiled",
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    detections=len(detections),
                    openvino_calls=1,
                    tiled_calls=1,
                )
                return detections
            detections = await asyncio.to_thread(
                self._detect_sync, frame, confidence_threshold, target_classes
            )
            self._record_metrics(
                path="openvino",
                latency_ms=(time.perf_counter() - start) * 1000.0,
                detections=len(detections),
                openvino_calls=1,
            )
            return detections

    async def _detect_coral(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        target_classes: Optional[list[str]],
    ) -> list[Detection]:
        """Run detection on Coral Edge TPU, fallback to OpenVINO on failure."""
        try:
            from services.coral_backend import coral_backend
            async with self._coral_lock:
                results = await asyncio.to_thread(
                    coral_backend.detect_objects, frame, confidence_threshold, target_classes
                )
            h, w = frame.shape[:2]
            frame_area = h * w
            detections = []
            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                cls = r["class_name"]
                conf = r["confidence"]

                # Per-class confidence thresholds (same rules as OpenVINO path)
                if cls == "person":
                    if conf < settings.detector_person_min_confidence:
                        continue
                    # Min size filter — 0.15% of frame (avoids filtering small children).
                    if (x2 - x1) * (y2 - y1) < frame_area * 0.0015:
                        continue
                elif cls in ("cat", "dog"):
                    if conf < settings.detector_animal_min_confidence:
                        continue
                else:
                    if conf < confidence_threshold:
                        continue

                if cls == "person":
                    box_h, box_w = y2 - y1, x2 - x1
                    cy1 = max(0, y1 - int(box_h * 0.25))
                    cy2 = min(h, y2 + int(box_h * 0.10))
                    cx1 = max(0, x1 - int(box_w * 0.10))
                    cx2 = min(w, x2 + int(box_w * 0.10))
                    crop = frame[cy1:cy2, cx1:cx2].copy()
                else:
                    crop = r["crop"]
                detections.append(Detection(cls, conf, r["bbox"], crop))

            detections = self._merge_contained_persons(detections)
            detections = self._fix_confused_classes(detections, frame)

            # If we're looking for animals and Coral found none, try OpenVINO.
            # Use single-pass YOLO here (the tracker's enhanced tiled scan provides
            # high-resolution coverage on a 5-15s interval separately).
            animal_targets = set(target_classes or []) & {"cat", "dog"}
            has_animals = any(d.class_name in ("cat", "dog") for d in detections)
            if animal_targets and not has_animals and self._initialized:
                async with self._semaphore:
                    ov_dets = await asyncio.to_thread(
                        self._detect_sync, frame,
                        settings.detector_animal_min_confidence,
                        list(animal_targets),
                    )
                self._record_metrics(
                    path="coral-animal-fallback",
                    latency_ms=0.0,
                    detections=len(ov_dets),
                    count_as_call=False,
                    coral_animal_fallbacks=1,
                )
                detections.extend(ov_dets)

            return detections
        except Exception as e:
            logger.warning("Coral detection failed, falling back to OpenVINO: %s", e)
            self._record_metrics(
                path="coral-error-fallback",
                latency_ms=0.0,
                detections=0,
                count_as_call=False,
                coral_fallback_errors=1,
            )
            if not self._initialized:
                return []
            async with self._semaphore:
                return await asyncio.to_thread(
                    self._detect_sync, frame, confidence_threshold, target_classes
                )

    async def _detect_remote(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        target_classes: Optional[list[str]],
    ) -> list[Detection]:
        """Offload detection to remote ML server."""
        try:
            from services.ml_client import remote_detect
            results = await remote_detect(frame, confidence_threshold, target_classes)
            detections = []
            h, w = frame.shape[:2]
            for r in results:
                bbox = tuple(r["bbox"]) if isinstance(r["bbox"], (list, tuple)) else r["bbox"]
                # Re-crop from original frame for consistent quality
                x1, y1, x2, y2 = bbox
                cls = r["class_name"]
                if cls == "person":
                    box_h, box_w = y2 - y1, x2 - x1
                    cy1 = max(0, y1 - int(box_h * 0.25))
                    cy2 = min(h, y2 + int(box_h * 0.10))
                    cx1 = max(0, x1 - int(box_w * 0.10))
                    cx2 = min(w, x2 + int(box_w * 0.10))
                    crop = frame[cy1:cy2, cx1:cx2].copy()
                else:
                    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)].copy()
                det = Detection(cls, r["confidence"], bbox, crop)
                detections.append(det)
            return self._fix_confused_classes(detections, frame)
        except Exception as e:
            logger.warning("Remote ML detect failed, falling back to local: %s", e)
            self._record_metrics(
                path="remote-error-fallback",
                latency_ms=0.0,
                detections=0,
                count_as_call=False,
                remote_fallback_errors=1,
            )
            if not self._initialized:
                return []
            async with self._semaphore:
                return await asyncio.to_thread(
                    self._detect_sync, frame, confidence_threshold, target_classes
                )

    def _detect_sync(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        target_classes: Optional[list[str]],
    ) -> list[Detection]:
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        result = self._compiled_model([blob])
        outputs = result[self._output_layer]

        # Seg model: extract prototype masks
        protos = None
        if self._is_seg and self._proto_layer is not None:
            protos = result[self._proto_layer]

        return self._postprocess(
            outputs, frame, scale, pad_x, pad_y,
            confidence_threshold, target_classes, protos,
        )

    def _detect_tiled(
        self,
        frame: np.ndarray,
        confidence_threshold: float,
        target_classes: Optional[list[str]],
    ) -> list[Detection]:
        """Run YOLO on 4 overlapping quadrant tiles for ~2x effective resolution.

        Each tile is ~half the frame, giving small objects ~4x more pixels in
        the YOLO 640x640 input.  Results are merged with NMS dedup.
        """
        h, w = frame.shape[:2]
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
            tile_dets = self._detect_sync(crop, confidence_threshold, target_classes)

            # Remap bbox from tile coords → full-frame coords
            for d in tile_dets:
                bx1, by1, bx2, by2 = d.bbox
                fx1 = max(0, bx1 + tx1)
                fy1 = max(0, by1 + ty1)
                fx2 = min(w, bx2 + tx1)
                fy2 = min(h, by2 + ty1)
                if fx2 > fx1 and fy2 > fy1:
                    full_crop = frame[fy1:fy2, fx1:fx2].copy()
                    all_dets.append(Detection(d.class_name, d.confidence, (fx1, fy1, fx2, fy2), full_crop))

        # NMS to deduplicate detections in overlap regions
        if len(all_dets) > 1:
            boxes = np.array([(d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]) for d in all_dets], dtype=np.float32)
            scores = np.array([d.confidence for d in all_dets], dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, 0.45)
            if len(indices) > 0:
                all_dets = [all_dets[i] for i in indices.flatten()]
            else:
                all_dets = []

        return all_dets

    async def stop(self):
        self._compiled_model = None
        self._initialized = False
        logger.info("Object detector stopped")


# Singleton
object_detector = ObjectDetector()
