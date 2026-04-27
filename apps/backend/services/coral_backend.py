"""BanusNas — Coral Edge TPU Backend: TFLite INT8 inference for YOLO + CNN.

The Coral USB Accelerator can only hold ONE model at a time in TPU SRAM.
Model swaps take ~50-100ms.  Strategy:
  - YOLO detection (primary workload) stays loaded most of the time.
  - CNN feature extraction swaps in on demand, then swaps YOLO back.
  - A threading.Lock serialises all TPU access.

Requires:
  - libedgetpu1-std (Debian/Ubuntu apt package)
  - tflite-runtime >= 2.14 (pip)
  - INT8 Edge TPU-compiled models (*_edgetpu.tflite)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# SSD MobileNet V2 COCO uses 1-indexed 91-class IDs (with gaps).
# Map directly from SSD class ID → COCO name.
COCO_SSD_LABELS: dict[int, str] = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}


class CoralBackend:
    """Thread-safe Coral Edge TPU inference for object detection and CNN features."""

    def __init__(self):
        self._lock = threading.Lock()
        self._yolo_interpreter = None
        self._cnn_interpreter = None
        self._active_model: Optional[str] = None  # "yolo" | "cnn" | None
        self._delegate = None
        self._available = False

        # Model paths (set from config)
        self._yolo_model_path: Optional[str] = None
        self._cnn_model_path: Optional[str] = None

        # YOLO model metadata
        self._yolo_input_size = 320  # INT8 models typically use 320×320
        self._yolo_num_classes = 80

        # CNN model metadata
        self._cnn_input_size = 224
        self._cnn_embed_dim = 1280

        # Quantisation parameters (read from model tensors)
        self._yolo_input_scale = 1.0
        self._yolo_input_zero_point = 0
        self._cnn_input_scale = 1.0
        self._cnn_input_zero_point = 0

        # Stats
        self._swap_count = 0
        self._last_swap_ms = 0.0
        self._detect_count = 0
        self._detect_total_ms = 0.0
        self._detect_last_ms = 0.0
        self._cnn_count = 0
        self._cnn_total_ms = 0.0
        self._cnn_last_ms = 0.0
        self._start_time: Optional[float] = None
        self._last_error: Optional[str] = None

    # ──────── Lifecycle ────────

    def start(self, yolo_model_path: str = "", cnn_model_path: str = "") -> bool:
        """Initialise the Edge TPU delegate and load available models.

        Either model path can be empty/missing — at least one valid model is required.
        CNN-only mode is fully supported (Frigate handles detection).
        """
        self._yolo_model_path = yolo_model_path
        self._cnn_model_path = cnn_model_path

        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                logger.warning("ai-edge-litert / tflite-runtime not installed — Coral backend disabled")
                return False

        # Try to load the Edge TPU delegate
        try:
            self._delegate = load_delegate("libedgetpu.so.1")
        except (ValueError, OSError) as e:
            logger.warning("Edge TPU delegate not available: %s — Coral disabled", e)
            return False

        # Check which models are available
        yolo_ok = bool(yolo_model_path) and Path(yolo_model_path).exists() and Path(yolo_model_path).stat().st_size > 1000
        cnn_ok = bool(cnn_model_path) and Path(cnn_model_path).exists() and Path(cnn_model_path).stat().st_size > 1000

        if not yolo_ok and not cnn_ok:
            logger.warning("Coral: no valid models found (YOLO: %s, CNN: %s) — disabled",
                           yolo_model_path or "(none)", cnn_model_path or "(none)")
            return False

        # Load whichever model is available (prefer CNN if both present, since
        # Frigate handles detection and CNN feature extraction is our primary use)
        try:
            if cnn_ok:
                self._cnn_interpreter = self._make_interpreter(cnn_model_path)
                self._read_cnn_metadata()
                self._active_model = "cnn"

                # Smoke-test CNN
                inp_detail = self._cnn_interpreter.get_input_details()[0]
                dummy = np.zeros(inp_detail["shape"], dtype=inp_detail["dtype"])
                self._cnn_interpreter.set_tensor(inp_detail["index"], dummy)
                self._cnn_interpreter.invoke()

                logger.info("Coral CNN loaded: %d×%d → %d-dim features",
                            self._cnn_input_size, self._cnn_input_size, self._cnn_embed_dim)
            elif yolo_ok:
                self._yolo_interpreter = self._make_interpreter(yolo_model_path)
                self._read_yolo_metadata()
                self._active_model = "yolo"

                # Smoke-test YOLO
                inp_detail = self._yolo_interpreter.get_input_details()[0]
                dummy = np.zeros(inp_detail["shape"], dtype=inp_detail["dtype"])
                self._yolo_interpreter.set_tensor(inp_detail["index"], dummy)
                self._yolo_interpreter.invoke()

            self._available = True
            self._start_time = time.time()
            logger.info(
                "Coral Edge TPU ready — YOLO: %s, CNN: %s",
                f"{self._yolo_input_size}×{self._yolo_input_size}" if yolo_ok else "disabled",
                f"{self._cnn_embed_dim}-dim" if cnn_ok else "disabled",
            )
            return True
        except Exception as e:
            logger.error("Coral Edge TPU init failed: %s", e)
            self._last_error = str(e)
            return False

    def _make_interpreter(self, model_path: str):
        """Create a new TFLite interpreter reusing the shared Edge TPU delegate."""
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter

        interp = Interpreter(
            model_path=model_path,
            experimental_delegates=[self._delegate],
        )
        interp.allocate_tensors()
        return interp

    def _read_yolo_metadata(self):
        """Read input/output shapes and quant params from the YOLO model."""
        inp = self._yolo_interpreter.get_input_details()[0]
        self._yolo_input_size = inp["shape"][1]  # [1, H, W, 3]
        self._yolo_input_scale = inp["quantization"][0]
        self._yolo_input_zero_point = inp["quantization"][1]
        out = self._yolo_interpreter.get_output_details()
        self._yolo_num_outputs = len(out)
        logger.info(
            "Coral YOLO: input %s quant=(%.6f, %d), %d outputs: %s",
            inp["shape"].tolist(),
            self._yolo_input_scale,
            self._yolo_input_zero_point,
            len(out),
            [o["shape"].tolist() for o in out],
        )

    def _read_cnn_metadata(self):
        """Read input shape and quant params from the CNN model."""
        inp = self._cnn_interpreter.get_input_details()[0]
        self._cnn_input_size = inp["shape"][1]
        self._cnn_input_scale = inp["quantization"][0]
        self._cnn_input_zero_point = inp["quantization"][1]
        out = self._cnn_interpreter.get_output_details()[0]
        self._cnn_embed_dim = out["shape"][-1]
        logger.info(
            "Coral CNN: input %s quant=(%.6f, %d), output dim=%d",
            inp["shape"].tolist(),
            self._cnn_input_scale,
            self._cnn_input_zero_point,
            self._cnn_embed_dim,
        )

    def _swap_to(self, target: str):
        """Swap the active model on the TPU.  Must be called with _lock held."""
        if self._active_model == target:
            return

        t0 = time.monotonic()

        if target == "yolo":
            if not self._yolo_model_path or not Path(self._yolo_model_path).exists():
                raise RuntimeError("YOLO model not available for Coral")
            self._cnn_interpreter = None  # release TPU
            self._yolo_interpreter = self._make_interpreter(self._yolo_model_path)
            self._read_yolo_metadata()
        elif target == "cnn":
            if not self._cnn_model_path or not Path(self._cnn_model_path).exists():
                raise RuntimeError("CNN model not available for Coral")
            self._yolo_interpreter = None  # release TPU
            self._cnn_interpreter = self._make_interpreter(self._cnn_model_path)
            self._read_cnn_metadata()

        self._active_model = target
        self._swap_count += 1
        self._last_swap_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "Coral model swap → %s  (%.1f ms, total swaps: %d)",
            target,
            self._last_swap_ms,
            self._swap_count,
        )

    # ──────── Object Detection ────────

    def detect_objects(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        target_classes: Optional[list[str]] = None,
    ) -> list[dict]:
        """Run YOLO detection on Coral.

        Returns list of dicts: {class_name, confidence, bbox: (x1,y1,x2,y2), crop}.
        """
        if not self._available:
            return []
        if not self._yolo_model_path or not Path(self._yolo_model_path).exists():
            return []  # CNN-only mode — no detection capability

        t0 = time.monotonic()
        try:
            from services.object_detector import COCO_CLASSES

            h, w = frame.shape[:2]
            sz = self._yolo_input_size

            with self._lock:
                self._swap_to("yolo")

                # Preprocess: resize with letterbox to sz×sz, quantize to uint8
                blob, scale, pad_x, pad_y = self._letterbox(frame, sz)
                quantized = self._quantize_input(
                    blob, self._yolo_input_scale, self._yolo_input_zero_point
                )

                # Run inference
                inp_detail = self._yolo_interpreter.get_input_details()[0]
                self._yolo_interpreter.set_tensor(inp_detail["index"], quantized)
                self._yolo_interpreter.invoke()

                # Read all outputs
                out_details = self._yolo_interpreter.get_output_details()
                outputs = []
                for od in out_details:
                    raw = self._yolo_interpreter.get_tensor(od["index"])
                    # Dequantize if INT8
                    if od["dtype"] == np.uint8 or od["dtype"] == np.int8:
                        s, zp = od["quantization"]
                        raw = (raw.astype(np.float32) - zp) * s
                    outputs.append(raw)

            # Postprocess outside the lock
            result = self._postprocess_yolo(
                outputs, frame, h, w, scale, pad_x, pad_y,
                confidence_threshold, target_classes, COCO_CLASSES,
            )
            elapsed = (time.monotonic() - t0) * 1000
            self._detect_count += 1
            self._detect_total_ms += elapsed
            self._detect_last_ms = elapsed
            return result
        except Exception as e:
            self._last_error = str(e)
            raise

    @staticmethod
    def _letterbox(frame: np.ndarray, target_size: int):
        """Resize frame with letterbox padding to target_size×target_size.

        Returns (blob_float32_NHWC, scale, pad_x, pad_y).
        """
        h, w = frame.shape[:2]
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # NHWC float32 [0,1]
        blob = canvas.astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, pad_x, pad_y

    @staticmethod
    def _quantize_input(
        blob_float: np.ndarray, scale: float, zero_point: int
    ) -> np.ndarray:
        """Convert float32 [0,1] blob to quantized uint8 for Edge TPU."""
        quantized = (blob_float / scale) + zero_point
        return np.clip(quantized, 0, 255).astype(np.uint8)

    @staticmethod
    def _postprocess_yolo(
        outputs: list[np.ndarray],
        frame: np.ndarray,
        orig_h: int,
        orig_w: int,
        scale: float,
        pad_x: int,
        pad_y: int,
        conf_threshold: float,
        target_classes: Optional[list[str]],
        class_names: list[str],
    ) -> list[dict]:
        """Parse YOLO TFLite outputs into detections.

        Handles multiple output formats:
        A) Ultralytics Edge TPU (4 outputs): boxes [1,N,4], classes [1,N], scores [1,N], count [1]
        B) Single output [1, N, 4+num_classes] or [1, 4+num_classes, N] (standard raw YOLO)
        """
        # --- Format A: Ultralytics multi-output (post-NMS) ---
        if len(outputs) >= 4:
            return CoralBackend._postprocess_yolo_multioutput(
                outputs, frame, orig_h, orig_w, scale, pad_x, pad_y,
                conf_threshold, target_classes, class_names,
            )

        # --- Format B: Single raw output (needs NMS) ---
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]  # remove batch dim → [N, C] or [C, N]

        # Auto-detect transposed format
        if raw.shape[0] < raw.shape[1] and raw.shape[0] == (4 + len(class_names)):
            raw = raw.T  # [C, N] → [N, C]

        n_cols = raw.shape[1]
        n_classes = n_cols - 4
        if n_classes <= 0:
            return []

        # Extract boxes (cx, cy, w, h) and class scores
        boxes = raw[:, :4]
        scores = raw[:, 4:]

        # Best class per detection
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by confidence
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert cx,cy,w,h → x1,y1,x2,y2 in letterbox coords
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Unscale from letterbox to original frame coords
        x1 = ((x1 - pad_x) / scale).astype(int)
        y1 = ((y1 - pad_y) / scale).astype(int)
        x2 = ((x2 - pad_x) / scale).astype(int)
        y2 = ((y2 - pad_y) / scale).astype(int)

        # Clip to frame bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        # NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=list(zip(x1.tolist(), y1.tolist(), (x2 - x1).tolist(), (y2 - y1).tolist())),
            scores=confidences.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=0.45,
        )
        if len(indices) == 0:
            return []
        indices = indices.flatten()

        results = []
        for i in indices:
            cid = int(class_ids[i])
            if cid >= len(class_names):
                continue
            cls = class_names[cid]
            if target_classes and cls not in target_classes:
                continue

            bx1, by1, bx2, by2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
            if bx2 <= bx1 or by2 <= by1:
                continue

            crop = frame[by1:by2, bx1:bx2].copy()
            results.append({
                "class_name": cls,
                "confidence": float(confidences[i]),
                "bbox": (bx1, by1, bx2, by2),
                "crop": crop,
            })

        return results

    @staticmethod
    def _postprocess_yolo_multioutput(
        outputs: list[np.ndarray],
        frame: np.ndarray,
        orig_h: int,
        orig_w: int,
        scale: float,
        pad_x: int,
        pad_y: int,
        conf_threshold: float,
        target_classes: Optional[list[str]],
        class_names: list[str],
    ) -> list[dict]:
        """Parse SSD / Ultralytics Edge TPU multi-output format (post-NMS).

        Outputs: [boxes (1,N,4), classes (1,N), scores (1,N), count (1)]
        Boxes are in y1,x1,y2,x2 normalised [0,1] format.
        Handles both SSD 1-indexed COCO-91 IDs and YOLO 0-indexed COCO-80 IDs.
        """
        sz = int(max(outputs[0].shape))  # infer input size from box count
        boxes_raw = outputs[0].squeeze()   # [N, 4] — y1,x1,y2,x2 normalised
        class_ids = outputs[1].squeeze()   # [N]
        scores_raw = outputs[2].squeeze()  # [N]
        count = int(outputs[3].squeeze()) if outputs[3].size == 1 else len(scores_raw)

        results = []
        input_sz = float(max(orig_h, orig_w))  # for denormalization

        for i in range(min(count, len(scores_raw))):
            conf = float(scores_raw[i])
            if conf < conf_threshold:
                continue

            cid = int(class_ids[i])
            # SSD COCO models use 1-indexed IDs; fall back to 0-indexed list
            cls = COCO_SSD_LABELS.get(cid)
            if cls is None and 0 <= cid < len(class_names):
                cls = class_names[cid]
            if cls is None:
                continue
            if target_classes and cls not in target_classes:
                continue

            # Boxes may be in different formats depending on Ultralytics version
            box = boxes_raw[i]
            if np.max(box) <= 1.0:
                # Normalised y1,x1,y2,x2 — scale to letterbox pixel coords
                y1_lb = box[0] * (max(orig_h, orig_w) * scale)
                x1_lb = box[1] * (max(orig_h, orig_w) * scale)
                y2_lb = box[2] * (max(orig_h, orig_w) * scale)
                x2_lb = box[3] * (max(orig_h, orig_w) * scale)
            else:
                # Already pixel coords in letterbox space
                y1_lb, x1_lb, y2_lb, x2_lb = box[0], box[1], box[2], box[3]

            # Unscale from letterbox to original frame
            bx1 = int(np.clip((x1_lb - pad_x) / scale, 0, orig_w))
            by1 = int(np.clip((y1_lb - pad_y) / scale, 0, orig_h))
            bx2 = int(np.clip((x2_lb - pad_x) / scale, 0, orig_w))
            by2 = int(np.clip((y2_lb - pad_y) / scale, 0, orig_h))

            if bx2 <= bx1 or by2 <= by1:
                continue

            crop = frame[by1:by2, bx1:bx2].copy()
            results.append({
                "class_name": cls,
                "confidence": conf,
                "bbox": (bx1, by1, bx2, by2),
                "crop": crop,
            })

        return results

    # ──────── CNN Feature Extraction ────────

    def extract_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract CNN features using the Coral TPU (MobileNetV2 INT8).

        Returns L2-normalised float32 embedding, or None if unavailable.
        """
        if not self._available or not self._cnn_model_path:
            return None
        cnn_path = Path(self._cnn_model_path)
        if not cnn_path.exists() or cnn_path.stat().st_size < 1000:
            return None

        t0 = time.monotonic()
        sz = self._cnn_input_size

        # Preprocess: resize to 224×224, ImageNet normalise, quantize
        resized = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0

        # ImageNet normalisation
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = (blob - mean) / std

        # NHWC shape for TFLite
        blob = np.expand_dims(blob, axis=0)

        # Quantize
        quantized = self._quantize_input(
            blob, self._cnn_input_scale, self._cnn_input_zero_point
        )

        with self._lock:
            self._swap_to("cnn")

            inp_detail = self._cnn_interpreter.get_input_details()[0]
            self._cnn_interpreter.set_tensor(inp_detail["index"], quantized)
            self._cnn_interpreter.invoke()

            out_detail = self._cnn_interpreter.get_output_details()[0]
            raw = self._cnn_interpreter.get_tensor(out_detail["index"])

            # Dequantize
            if out_detail["dtype"] == np.uint8 or out_detail["dtype"] == np.int8:
                s, zp = out_detail["quantization"]
                raw = (raw.astype(np.float32) - zp) * s

        # L2 normalise
        embedding = raw.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        elapsed = (time.monotonic() - t0) * 1000
        self._cnn_count += 1
        self._cnn_total_ms += elapsed
        self._cnn_last_ms = elapsed

        return embedding

    # ──────── Diagnostics ────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def cnn_available(self) -> bool:
        cnn_path = Path(self._cnn_model_path) if self._cnn_model_path else None
        return (
            self._available
            and cnn_path is not None
            and cnn_path.exists()
            and cnn_path.stat().st_size > 1000
        )

    def status(self) -> dict:
        uptime = round(time.time() - self._start_time) if self._start_time else 0
        detect_avg = round(self._detect_total_ms / self._detect_count, 1) if self._detect_count else 0
        cnn_avg = round(self._cnn_total_ms / self._cnn_count, 1) if self._cnn_count else 0
        return {
            "available": bool(self._available),
            "active_model": self._active_model,
            "swap_count": int(self._swap_count),
            "last_swap_ms": round(self._last_swap_ms, 1),
            "yolo_input_size": int(self._yolo_input_size),
            "cnn_embed_dim": int(self._cnn_embed_dim) if self.cnn_available else None,
            "detect_count": int(self._detect_count),
            "detect_avg_ms": float(detect_avg),
            "detect_last_ms": round(self._detect_last_ms, 1),
            "cnn_count": int(self._cnn_count),
            "cnn_avg_ms": float(cnn_avg),
            "cnn_last_ms": round(self._cnn_last_ms, 1),
            "uptime_seconds": int(uptime),
            "last_error": self._last_error,
        }


# Singleton
coral_backend = CoralBackend()
