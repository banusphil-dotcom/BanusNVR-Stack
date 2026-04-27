"""BanusNas — Recognition Service: MobileNetV2 CNN + Person ReID.

Person re-identification model (OpenVINO person-reidentification-retail-0287,
ResNet50-based, 256-dim) for body-based person matching when face is not visible.
MobileNetV2 is retained for pet/vehicle/other matching.
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# MobileNetV2 ONNX — exported from torchvision, features-only (no classifier head)
_MOBILENET_MODEL = "/models/mobilenetv2_features.onnx"
CNN_EMBED_DIM = 1280

# Person Re-ID model: OpenVINO's person-reidentification-retail-0287 (ResNet50-based)
# Input: [1, 3, 256, 128] NCHW BGR float32 [0,1]   Output: [1, 256]
_REID_MODEL_DIR = Path("/models/person_reid")
_REID_XML = "person-reidentification-retail-0287.xml"
_REID_BIN = "person-reidentification-retail-0287.bin"
_REID_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/"
    "2023.0/models_bin/1/person-reidentification-retail-0287/FP16"
)
REID_EMBED_DIM = 256


class RecognitionResult:
    """Result of a recognition attempt."""

    def __init__(self, subject: Optional[str], confidence: float, subject_id: Optional[str] = None, margin: float = 0.0):
        self.subject = subject  # Named object name or None
        self.confidence = confidence
        self.subject_id = subject_id
        self.margin = margin  # Gap between best and second-best score


class RecognitionService:
    """Pet/object matching via MobileNetV2, person body matching via ReID."""

    def __init__(self):
        # CNN feature extractor (MobileNetV2 via OpenVINO)
        self._cnn_model = None
        self._cnn_input = None
        self._cnn_lock = threading.Lock()
        self._cnn_output = None
        self._cnn_ready = False
        # Person ReID model (OpenVINO)
        self._reid_model = None
        self._reid_input = None
        self._reid_output = None
        self._reid_ready = False
        self._reid_lock = threading.Lock()

    async def start(self):
        # Load CNN feature extractor
        await self._load_cnn_model()
        # Load person ReID model
        await self._load_reid_model()
        logger.info(
            "RecognitionService started — CNN: %s, ReID: %s",
            "ready" if self._cnn_ready else "disabled",
            "ready" if self._reid_ready else "disabled",
        )

    async def _load_cnn_model(self):
        """Load MobileNetV2 feature extractor via OpenVINO."""
        model_path = Path(_MOBILENET_MODEL)
        if not model_path.exists():
            logger.warning("CNN feature model not found at %s — pet matching will use histograms", model_path)
            return
        try:
            await asyncio.to_thread(self._load_cnn_sync, str(model_path))
        except Exception as e:
            logger.error("Failed to load CNN feature model: %s", e)

    def _load_cnn_sync(self, model_path: str):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(model_path)
        # Run on CPU to keep GPU free for YOLO
        self._cnn_model = core.compile_model(
            model, "CPU",
            {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "2"},
        )
        self._cnn_input = self._cnn_model.input(0)
        self._cnn_output = self._cnn_model.output(0)
        self._cnn_ready = True
        logger.info("CNN feature extractor loaded (MobileNetV2, %d-dim, CPU)", CNN_EMBED_DIM)

    # ────────── Person ReID model ──────────

    async def _load_reid_model(self):
        """Load person ReID model (OpenVINO IR), downloading if missing."""
        xml_path = _REID_MODEL_DIR / _REID_XML
        bin_path = _REID_MODEL_DIR / _REID_BIN
        if not xml_path.exists() or not bin_path.exists():
            await self._download_reid(xml_path, bin_path)
        if not xml_path.exists() or not bin_path.exists():
            logger.warning("Person ReID model not available — body-based person matching disabled")
            return
        try:
            await asyncio.to_thread(self._load_reid_sync, str(xml_path))
        except Exception as e:
            logger.error("Failed to load person ReID model: %s", e)

    def _load_reid_sync(self, xml_path: str):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(xml_path)
        self._reid_model = core.compile_model(
            model, "CPU",
            {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": "2"},
        )
        self._reid_input = self._reid_model.input(0)
        self._reid_output = self._reid_model.output(0)
        self._reid_ready = True
        logger.info("Person ReID loaded (person-reidentification-retail-0287, %d-dim, CPU)", REID_EMBED_DIM)

    @staticmethod
    async def _download_reid(xml_path: Path, bin_path: Path):
        """Download OpenVINO person-reidentification-retail-0287 FP16 model."""
        _REID_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
            for fname, dest in [(_REID_XML, xml_path), (_REID_BIN, bin_path)]:
                url = f"{_REID_BASE_URL}/{fname}"
                try:
                    logger.info("Downloading person ReID model: %s", fname)
                    r = await client.get(url)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    logger.info("Saved %s (%d bytes)", fname, len(r.content))
                except Exception as e:
                    logger.error("ReID model download failed for %s: %s", fname, e)

    async def stop(self):
        pass

    # ────────── CNN feature extraction (MobileNetV2) ──────────

    def _compute_cnn_embedding(self, crop: np.ndarray) -> list[float]:
        """Compute a 1280-dim L2-normalised feature vector via MobileNetV2."""
        # ImageNet preprocessing: resize to 224×224, normalise with ImageNet stats
        resized = cv2.resize(crop, (224, 224))
        blob = resized.astype(np.float32) / 255.0
        blob = (blob - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
               np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # HWC BGR → RGB → CHW → NCHW
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)

        with self._cnn_lock:
            result = self._cnn_model([blob])
        features = result[self._cnn_output].flatten().astype(np.float64)
        # L2-normalise
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.tolist()

    def _compute_histogram_embedding(self, crop: np.ndarray) -> list[float]:
        """Legacy: spatial color histogram embedding (fallback if CNN unavailable)."""
        resized = cv2.resize(crop, (128, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        h, w = resized.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            hsv[0:mid_h, 0:mid_w],
            hsv[0:mid_h, mid_w:w],
            hsv[mid_h:h, 0:mid_w],
            hsv[mid_h:h, mid_w:w],
        ]

        all_hists: list[np.ndarray] = []
        for quad in quadrants:
            for channel, bins, range_ in [(0, 16, [0, 180]), (1, 16, [0, 256]), (2, 16, [0, 256])]:
                hist = cv2.calcHist([quad], [channel], None, [bins], range_)
                cv2.normalize(hist, hist)
                all_hists.append(hist.flatten())

        return np.concatenate(all_hists).tolist()

    def _compute_embedding_best(self, crop: np.ndarray) -> list[float]:
        """Compute embedding using Coral TPU, OpenVINO CNN, or histogram (priority order)."""
        # 1. Coral Edge TPU (fastest — ~10ms on TPU)
        from core.config import settings
        if settings.coral_enabled:
            try:
                from services.coral_backend import coral_backend
                if coral_backend.cnn_available:
                    embedding = coral_backend.extract_features(crop)
                    if embedding is not None and len(embedding) > 0:
                        return embedding.tolist()
            except Exception as e:
                logger.debug("Coral CNN fallback: %s", e)
        # 2. OpenVINO CPU CNN
        if self._cnn_ready:
            return self._compute_cnn_embedding(crop)
        return self._compute_histogram_embedding(crop)

    async def match_pet(
        self, crop: np.ndarray, known_pets: list[tuple[int, str, list[float]]],
        exclude_ids: set[int] | None = None,
    ) -> Optional[RecognitionResult]:
        """Match a crop against known embeddings using cosine similarity.

        *exclude_ids* — set of object IDs to skip (e.g. vetoed by agent).
        """
        ranked = await self.match_pet_ranked(crop, known_pets, exclude_ids)
        if not ranked:
            return None
        return ranked[0]

    async def match_pet_ranked(
        self, crop: np.ndarray, known_pets: list[tuple[int, str, list[float]]],
        exclude_ids: set[int] | None = None,
    ) -> list[RecognitionResult]:
        """Score a crop against ALL known pet embeddings, return ranked list.

        Returns list of ``RecognitionResult`` sorted by confidence descending.
        Only candidates above the similarity threshold are included.
        ``margin`` is the gap between this candidate and the next-best.
        """
        if not known_pets:
            return []

        # Remote offload for embedding computation
        from services.ml_client import ml_offload_enabled
        if ml_offload_enabled:
            try:
                from services.ml_client import remote_embedding
                embedding = await remote_embedding(crop, model="cnn")
            except Exception as e:
                logger.warning("Remote CNN embedding failed in match_pet, falling back: %s", e)
                embedding = self._compute_embedding_best(crop)
        else:
            embedding = self._compute_embedding_best(crop)
        emb_array = np.array(embedding, dtype=np.float64)
        emb_dim = len(emb_array)

        # Score ALL candidates
        candidates: list[tuple[int, str, float]] = []
        for obj_id, name, stored_emb in known_pets:
            if exclude_ids and obj_id in exclude_ids:
                continue
            if not stored_emb:
                continue
            stored_array = np.array(stored_emb, dtype=np.float64)
            if len(stored_array) != emb_dim:
                continue  # Dimension mismatch (old histogram vs new CNN) — skip
            dot = float(np.dot(emb_array, stored_array))
            norm = float(np.linalg.norm(emb_array) * np.linalg.norm(stored_array))
            similarity = dot / (norm + 1e-8)
            candidates.append((obj_id, name, similarity))

        if not candidates:
            return []

        candidates.sort(key=lambda c: c[2], reverse=True)

        # CNN features: threshold ~0.45 (lowered for variable-appearance pets like white cats).
        # For low-quality crops (e.g. Frigate's 300×300 fallback when HD path
        # fails) genuine matches often score 0.35–0.45. We additionally accept
        # candidates with score >= 0.30 IF they have a clear margin (>=0.06)
        # over the next candidate — those are statistically distinct, not noise.
        # The downstream colour gate + per-call threshold in _recognize_pet
        # still enforce strict acceptance criteria.
        # histogram: ~0.80
        threshold = 0.30 if emb_dim == CNN_EMBED_DIM else 0.80
        margin_floor = 0.06 if emb_dim == CNN_EMBED_DIM else 0.10

        results: list[RecognitionResult] = []
        for i, (obj_id, name, score) in enumerate(candidates):
            if score <= threshold:
                break
            next_score = candidates[i + 1][2] if i + 1 < len(candidates) else 0.0
            margin = score - next_score
            # When score is below the legacy 0.45 cutoff, require a real margin
            if score < 0.45 and i == 0 and margin < margin_floor:
                break
            results.append(RecognitionResult(
                subject=name,
                confidence=score,
                subject_id=str(obj_id),
                margin=margin,
            ))
        return results

    async def compute_embedding(self, crop: np.ndarray) -> list[float]:
        """Compute embedding for any crop (for clustering unnamed objects)."""
        from services.ml_client import ml_offload_enabled
        if ml_offload_enabled:
            try:
                from services.ml_client import remote_embedding
                return await remote_embedding(crop, model="cnn")
            except Exception as e:
                logger.warning("Remote CNN embedding failed, falling back to local: %s", e)
        return self._compute_embedding_best(crop)

    def compute_and_merge_embedding(
        self, crop: np.ndarray, existing_embedding: Optional[list[float]], count: int
    ) -> list[float]:
        """Compute embedding from crop and merge with running average."""
        new_emb = self._compute_embedding_best(crop)
        if not existing_embedding or count == 0:
            return new_emb

        old = np.array(existing_embedding, dtype=np.float64)
        new = np.array(new_emb, dtype=np.float64)
        if len(old) != len(new):
            return new_emb  # Dimension change — start fresh

        merged = (old * count + new) / (count + 1)
        # Re-normalise for CNN embeddings
        if len(merged) == CNN_EMBED_DIM:
            norm = np.linalg.norm(merged)
            if norm > 0:
                merged = merged / norm
        return merged.tolist()

    async def compute_and_merge_embedding_async(
        self, crop: np.ndarray, existing_embedding: Optional[list[float]], count: int
    ) -> list[float]:
        """Async version — uses remote ML offload when available for consistent embeddings."""
        new_emb = await self.compute_embedding(crop)
        if not existing_embedding or count == 0:
            return new_emb

        old = np.array(existing_embedding, dtype=np.float64)
        new = np.array(new_emb, dtype=np.float64)
        if len(old) != len(new):
            return new_emb

        merged = (old * count + new) / (count + 1)
        if len(merged) == CNN_EMBED_DIM:
            norm = np.linalg.norm(merged)
            if norm > 0:
                merged = merged / norm
        return merged.tolist()

    # ────────── Person ReID (body-based person matching) ──────────

    def compute_reid_embedding(self, crop: np.ndarray) -> Optional[list[float]]:
        """Compute 256-dim person ReID embedding from a person crop.

        Input: BGR person bounding-box crop.
        Returns L2-normalised 256-dim list, or None if ReID unavailable.
        """
        # Note: remote offload for ReID is handled via async compute_reid_embedding_async
        if not self._reid_ready or self._reid_model is None:
            return None
        # Model expects [1, 3, 256, 128] NCHW BGR float32 [0, 1]
        resized = cv2.resize(crop, (128, 256))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
        blob = np.expand_dims(blob, 0)  # NCHW
        with self._reid_lock:
            result = self._reid_model([blob])
        features = result[self._reid_output].flatten().astype(np.float64)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.tolist()

    def compute_reid_embedding_sync(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Synchronous ReID embedding returning numpy array (for tracker thread)."""
        if not self._reid_ready or self._reid_model is None:
            return None
        resized = cv2.resize(crop, (128, 256))
        blob = resized.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, 0)
        with self._reid_lock:
            result = self._reid_model([blob])
        features = result[self._reid_output].flatten().astype(np.float64)
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    async def compute_reid_embedding_async(self, crop: np.ndarray) -> Optional[list[float]]:
        """Async ReID — uses remote ML offload when available for consistent embeddings."""
        from services.ml_client import ml_offload_enabled
        if ml_offload_enabled:
            try:
                from services.ml_client import remote_embedding
                return await remote_embedding(crop, model="reid")
            except Exception as e:
                logger.warning("Remote ReID embedding failed, falling back to local: %s", e)
        return self.compute_reid_embedding(crop)

    async def match_person_body(
        self,
        crop: np.ndarray,
        known_persons: list[tuple[int, str, list[float]]],
        threshold: float = 0.45,
        exclude_ids: set[int] | None = None,
    ) -> Optional[RecognitionResult]:
        """Match a person body crop against known body embeddings using ReID.

        *known_persons*: list of (obj_id, name, body_embedding).
        *exclude_ids* — set of object IDs to skip (e.g. vetoed by agent).
        """
        ranked = await self.match_person_body_ranked(crop, known_persons, threshold, exclude_ids)
        if not ranked:
            return None
        return ranked[0]

    async def match_person_body_ranked(
        self,
        crop: np.ndarray,
        known_persons: list[tuple[int, str, list[float]]],
        threshold: float = 0.45,
        exclude_ids: set[int] | None = None,
    ) -> list[RecognitionResult]:
        """Score a person body crop against ALL known body embeddings, return ranked list.

        Returns list of ``RecognitionResult`` sorted by confidence descending.
        Only candidates above ``threshold`` are included.
        ``margin`` is the gap between this candidate and the next-best.
        """
        if not known_persons:
            return []

        # Try remote offload first
        from services.ml_client import ml_offload_enabled
        if ml_offload_enabled:
            try:
                from services.ml_client import remote_embedding
                emb_list = await remote_embedding(crop, model="reid")
            except Exception as e:
                logger.warning("Remote ReID failed, falling back to local: %s", e)
                emb_list = self.compute_reid_embedding(crop)
        else:
            if not self._reid_ready:
                return []
            emb_list = self.compute_reid_embedding(crop)
        if emb_list is None:
            return []

        emb = np.array(emb_list, dtype=np.float64)

        # Score ALL candidates
        candidates: list[tuple[int, str, float]] = []
        for obj_id, name, stored_emb in known_persons:
            if exclude_ids and obj_id in exclude_ids:
                continue
            if not stored_emb or len(stored_emb) != REID_EMBED_DIM:
                continue
            ref = np.array(stored_emb, dtype=np.float64)
            score = float(np.dot(emb, ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-10))
            candidates.append((obj_id, name, score))

        if not candidates:
            return []

        candidates.sort(key=lambda c: c[2], reverse=True)

        results: list[RecognitionResult] = []
        for i, (obj_id, name, score) in enumerate(candidates):
            if score < threshold:
                break
            next_score = candidates[i + 1][2] if i + 1 < len(candidates) else 0.0
            margin = score - next_score
            results.append(RecognitionResult(
                subject=name,
                confidence=score,
                subject_id=str(obj_id),
                margin=margin,
            ))
        return results

    def merge_reid_embedding(
        self, existing: Optional[list[float]], new: list[float], count: int
    ) -> list[float]:
        """Running-average merge of body ReID embeddings, L2-normalised."""
        new_arr = np.array(new, dtype=np.float64)
        if existing and len(existing) == REID_EMBED_DIM and count > 0:
            old_arr = np.array(existing, dtype=np.float64)
            merged = (old_arr * count + new_arr) / (count + 1)
        else:
            merged = new_arr
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        return merged.tolist()

    @property
    def reid_available(self) -> bool:
        return self._reid_ready

# Singleton
recognition_service = RecognitionService()
