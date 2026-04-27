"""
BanusNVR ML Server.

Implements the HTTP contract used by `backend/services/ml_client.py`:

    POST /v1/detect             { image (b64), confidence_threshold, target_classes? }
    POST /v1/embedding          { image (b64), model: "cnn" | "reid" }
    POST /v1/faces              { image (b64), min_face_size }
    POST /v1/chat/completions   OpenAI-compatible (proxied to LLM_BACKEND_URL)
    GET  /v1/health

This reference implementation uses CPU-only models (Ultralytics YOLO, InsightFace
buffalo_s, MobileNetV2 features). Replace with TensorRT / GPU equivalents for
production GPU offload.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("banusnvr-ml")
logging.basicConfig(level=logging.INFO)

API_KEY = os.environ.get("ML_API_KEY", "").strip()
LLM_BACKEND_URL = os.environ.get("LLM_BACKEND_URL", "").strip()

app = FastAPI(title="BanusNVR ML Server", version="1.0.0")

# Lazily-loaded singletons
_yolo = None
_face_app = None
_reid_session = None
_cnn_session = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid image")
    return img


def _encode(img: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    sent = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if sent != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


def _get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO  # type: ignore
        model_path = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
        logger.info("Loading YOLO model: %s", model_path)
        _yolo = YOLO(model_path)
    return _yolo


def _get_face():
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis  # type: ignore
        logger.info("Loading InsightFace buffalo_s")
        fa = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
        fa.prepare(ctx_id=-1, det_size=(640, 640))
        _face_app = fa
    return _face_app


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DetectRequest(BaseModel):
    image: str
    confidence_threshold: float = 0.5
    target_classes: Optional[list[str]] = None


class EmbeddingRequest(BaseModel):
    image: str
    model: str = "cnn"


class FacesRequest(BaseModel):
    image: str
    min_face_size: int = Field(default=20, ge=4)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/v1/health")
async def health():
    return {"status": "ok", "auth_required": bool(API_KEY)}


@app.post("/v1/detect")
async def detect(req: DetectRequest, request: Request):
    _check_auth(request)
    img = _decode(req.image)
    yolo = _get_yolo()

    results = yolo.predict(
        img, conf=req.confidence_threshold, verbose=False, imgsz=640
    )
    detections = []
    if not results:
        return {"detections": []}
    r = results[0]
    names = r.names
    boxes = r.boxes
    for i in range(len(boxes)):
        cls_idx = int(boxes.cls[i].item())
        cls_name = names[cls_idx]
        if req.target_classes and cls_name not in req.target_classes:
            continue
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i].tolist())
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        detections.append({
            "class_name": cls_name,
            "confidence": float(boxes.conf[i].item()),
            "bbox": [x1, y1, x2, y2],
            "crop": _encode(crop),
        })
    return {"detections": detections}


@app.post("/v1/embedding")
async def embedding(req: EmbeddingRequest, request: Request):
    _check_auth(request)
    img = _decode(req.image)
    # Reference: a 256-d random projection of resized pixel mean — replace with
    # a real ReID / MobileNetV2 ONNX model in production.
    resized = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    pooled = resized.reshape(-1, 3).mean(axis=0)  # 3 dims
    rng = np.random.default_rng(seed=int(pooled.sum() * 1e6) & 0xFFFFFFFF)
    vec = rng.standard_normal(256).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return {"embedding": vec.tolist(), "model": req.model}


@app.post("/v1/faces")
async def faces(req: FacesRequest, request: Request):
    _check_auth(request)
    img = _decode(req.image)
    fa = _get_face()
    found = fa.get(img)
    out = []
    for f in found:
        x1, y1, x2, y2 = (int(v) for v in f.bbox.tolist())
        if (x2 - x1) < req.min_face_size or (y2 - y1) < req.min_face_size:
            continue
        aligned_b64 = None
        if hasattr(f, "aligned") and f.aligned is not None:
            aligned_b64 = _encode(f.aligned)
        emb = f.normed_embedding.tolist() if hasattr(f, "normed_embedding") else None
        out.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(f.det_score),
            "embedding": emb,
            "aligned_crop": aligned_b64,
        })
    return {"faces": out}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """Optional OpenAI-compatible LLM proxy. Set LLM_BACKEND_URL to enable."""
    _check_auth(request)
    if not LLM_BACKEND_URL:
        raise HTTPException(status_code=503, detail="LLM backend not configured")
    body = await request.json()
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{LLM_BACKEND_URL.rstrip('/')}/v1/chat/completions",
            json=body,
            headers={"content-type": "application/json"},
        )
    return r.json()
