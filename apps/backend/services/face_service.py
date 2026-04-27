"""BanusNas — Face Service: InsightFace (SCRFD + ArcFace 512-dim + GenderAge).

Uses InsightFace's buffalo_s model pack:
  - Face detection:  SCRFD-500MF  (~2.5 MB, much better recall on small/profile faces)
  - Face recognition: MobileFaceNet ArcFace (512-dim, ~13 MB)
  - Gender/Age: GenderAge estimator (~1.3 MB) — for soft biometric attributes

Only 3D landmarks (137 MB) and 2D landmarks (4.8 MB) are skipped to save memory.
"""

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ArcFace produces 512-dim L2-normalised embeddings
FACE_EMBED_DIM = 512


class FaceResult:
    """A single detected face."""

    __slots__ = ("bbox", "score", "face_data", "embedding", "aligned_crop")

    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        score: float,
        face_data,
        embedding: Optional[list[float]] = None,
        aligned_crop: Optional[np.ndarray] = None,
    ):
        self.bbox = bbox  # (x1, y1, x2, y2) in source image
        self.score = score
        self.face_data = face_data  # InsightFace Face object (or legacy ndarray)
        self.embedding = embedding
        self.aligned_crop = aligned_crop


class FaceService:
    """Detect faces with SCRFD, extract 512-dim ArcFace embeddings via InsightFace."""

    def __init__(self):
        self._app = None  # insightface.app.FaceAnalysis
        self._models_dir = Path("/models/insightface")
        self._ready = False

    # ──────── lifecycle ────────

    async def start(self):
        """Load InsightFace model pack, downloading buffalo_s if missing."""
        try:
            await asyncio.to_thread(self._load)
            self._ready = True
            logger.info(
                "FaceService ready (InsightFace SCRFD + ArcFace %d-dim)",
                FACE_EMBED_DIM,
            )
        except Exception as e:
            logger.error("FaceService InsightFace init failed: %s", e)
            logger.info("FaceService disabled — install 'insightface' and 'onnxruntime' packages")

    def _load(self):
        # Suppress InsightFace's FutureWarning about deprecated 'estimate'
        warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

        from insightface.app import FaceAnalysis

        self._models_dir.mkdir(parents=True, exist_ok=True)
        # buffalo_s: SCRFD-500MF (~2.5 MB) + MobileFaceNet ArcFace (~13 MB) + GenderAge (~1.3 MB)
        # Only skip 3D landmarks (137 MB) and 2D landmarks (4.8 MB)
        self._app = FaceAnalysis(
            name="buffalo_s",
            root=str(self._models_dir),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition", "genderage"],
        )
        # det_size=(640,640) gives good coverage.  ctx_id=-1 = CPU
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    # ──────── face detection ────────

    async def detect_faces_async(
        self, image: np.ndarray, min_face_size: int = 20
    ) -> list[FaceResult]:
        """Async face detection — uses remote ML server when offload is enabled."""
        from services.ml_client import ml_offload_enabled
        if ml_offload_enabled:
            try:
                from services.ml_client import remote_detect_faces
                remote_faces = await remote_detect_faces(image, min_face_size)
                results = []
                for rf in remote_faces:
                    results.append(FaceResult(
                        bbox=rf["bbox"],
                        score=rf["score"],
                        face_data=None,
                        embedding=rf.get("embedding"),
                        aligned_crop=rf.get("aligned_crop"),
                    ))
                return results
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "Remote face detection failed, falling back to local: %s", e
                )
        return await asyncio.to_thread(self.detect_faces, image, min_face_size)

    def detect_faces(
        self, image: np.ndarray, min_face_size: int = 20
    ) -> list[FaceResult]:
        """Detect all faces in *image*, returned largest-first.

        Each FaceResult already contains a 512-dim ArcFace embedding.
        """
        if not self._ready or self._app is None:
            return []

        h, w = image.shape[:2]
        if h < min_face_size or w < min_face_size:
            return []

        faces = self._app.get(image)
        if not faces and (h < 400 or w < 400):
            # Small crop (typical Frigate thumbnail) — retry at 2× resolution
            # to give SCRFD enough pixels to detect the face.
            scale = 2.0
            up = cv2.resize(image, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_LINEAR)
            faces_up = self._app.get(up)
            if faces_up:
                # Rescale bbox & landmarks back to original coordinate space
                for f in faces_up:
                    f.bbox = (f.bbox / scale).astype(f.bbox.dtype)
                    if getattr(f, "kps", None) is not None:
                        f.kps = (f.kps / scale).astype(f.kps.dtype)
                faces = faces_up
        if not faces:
            return []

        results: list[FaceResult] = []
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            fw, fh = x2 - x1, y2 - y1
            if fw < min_face_size or fh < min_face_size:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            emb = None
            if face.embedding is not None:
                emb = face.normed_embedding.tolist()

            # Build 112×112 aligned crop for thumbnail
            aligned = self._get_aligned_crop(image, face)

            results.append(
                FaceResult(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    score=float(face.det_score),
                    face_data=face,
                    embedding=emb,
                    aligned_crop=aligned,
                )
            )

        # Largest face first (closest / most prominent)
        results.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )
        return results

    @staticmethod
    def _get_aligned_crop(image: np.ndarray, face) -> np.ndarray:
        """Get a 112×112 aligned face crop using InsightFace landmarks."""
        try:
            from insightface.utils.face_align import norm_crop

            if face.kps is not None and len(face.kps) >= 5:
                return norm_crop(image, face.kps)
        except Exception:
            pass
        # Fallback: simple bbox crop resized to 112×112
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return cv2.resize(crop, (112, 112))

    # ──────── face embedding ────────

    def compute_face_embedding(
        self, image: np.ndarray, face_data
    ) -> tuple[list[float], np.ndarray]:
        """Extract 512-dim ArcFace embedding from a detected face.

        *face_data* is the InsightFace Face object stored in FaceResult.
        Returns ``(embedding_list, aligned_crop_112x112)``.
        """
        if not self._ready:
            return [], np.array([])

        # InsightFace already computed the embedding during detect_faces()
        if hasattr(face_data, "normed_embedding") and face_data.normed_embedding is not None:
            aligned = self._get_aligned_crop(image, face_data)
            return face_data.normed_embedding.tolist(), aligned

        # Fallback: re-run detection on the image region
        faces = self._app.get(image) if self._app else []
        if faces:
            best = faces[0]
            aligned = self._get_aligned_crop(image, best)
            if best.normed_embedding is not None:
                return best.normed_embedding.tolist(), aligned

        return [], np.array([])

    # ──────── matching ────────

    def match_face(
        self,
        embedding: list[float],
        known: list[tuple[int, str, list[float]]],
        threshold: float = 0.15,
        exclude_ids: set[int] | None = None,
    ) -> Optional[tuple[str, int, float, float]]:
        """Match *embedding* against known person face embeddings.

        *known* is a list of ``(object_id, name, stored_embedding)``.
        *exclude_ids* — set of object IDs to skip (e.g. vetoed by agent).
        Returns ``(name, obj_id, cosine_score, margin)`` or ``None``.
        Low ``threshold`` acts as noise filter; the caller applies the real
        decision threshold after attribute multiplier.
        ``margin`` is best_score - second_best_score.
        """
        ranked = self.match_face_ranked(embedding, known, threshold, exclude_ids)
        if not ranked:
            return None
        best = ranked[0]
        return best

    def match_face_ranked(
        self,
        embedding: list[float],
        known: list[tuple[int, str, list[float]]],
        threshold: float = 0.15,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[str, int, float, float]]:
        """Score *embedding* against ALL known face embeddings, return ranked list.

        Returns list of ``(name, obj_id, cosine_score, margin)`` sorted by
        score descending.  Only candidates above ``threshold`` are included.
        ``margin`` is the gap between this candidate and the next-best.
        """
        if not embedding or not known:
            return []

        emb_dim = len(embedding)
        emb = np.array(embedding, dtype=np.float64)

        # Score ALL candidates, rank by similarity
        candidates: list[tuple[str, int, float]] = []
        for obj_id, name, stored in known:
            if exclude_ids and obj_id in exclude_ids:
                continue
            if not stored or len(stored) != emb_dim:
                continue
            ref = np.array(stored, dtype=np.float64)
            score = float(
                np.dot(emb, ref)
                / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-10)
            )
            candidates.append((name, obj_id, score))

        if not candidates:
            return []

        candidates.sort(key=lambda c: c[2], reverse=True)

        # Build ranked results with per-candidate margin
        results: list[tuple[str, int, float, float]] = []
        for i, (name, obj_id, score) in enumerate(candidates):
            if score < threshold:
                break  # All remaining are below threshold
            next_score = candidates[i + 1][2] if i + 1 < len(candidates) else 0.0
            margin = score - next_score
            results.append((name, obj_id, score, margin))
        return results

    # ──────── utilities ────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity (for use outside the service, e.g. audit)."""
        va = np.array(a, dtype=np.float64)
        vb = np.array(b, dtype=np.float64)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))

    @staticmethod
    def merge_face_embeddings(
        existing: Optional[list[float]], new: list[float], count: int
    ) -> list[float]:
        """Running-average merge of face embeddings, L2-normalised."""
        new_arr = np.array(new, dtype=np.float64)
        new_dim = len(new_arr)
        if existing and len(existing) == new_dim and count > 0:
            old_arr = np.array(existing, dtype=np.float64)
            # Check compatibility: if cosine is very low, models have changed
            cos = float(np.dot(old_arr, new_arr) / (np.linalg.norm(old_arr) * np.linalg.norm(new_arr) + 1e-10))
            if cos < 0.15:
                # Model change detected (e.g. buffalo_s → buffalo_l) — start fresh
                merged = new_arr
            else:
                merged = (old_arr * count + new_arr) / (count + 1)
        else:
            # Dimension mismatch (model upgrade) — start fresh with new embedding
            merged = new_arr
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        return merged.tolist()

    @property
    def is_available(self) -> bool:
        from services.ml_client import ml_offload_enabled
        return self._ready or ml_offload_enabled


# Singleton
face_service = FaceService()
