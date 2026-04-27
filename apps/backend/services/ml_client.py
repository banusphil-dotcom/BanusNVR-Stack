"""BanusNas — Remote ML Client for offloading inference to a GPU server."""

import asyncio
import base64
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Module-level state — toggled by _apply_performance_settings
ml_offload_enabled: bool = False
ml_offload_url: str = "https://ml.banusphotos.com"

_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

# Cached health status to avoid per-event health checks
_health_ok: bool = False
_health_checked_at: float = 0.0
_HEALTH_CACHE_TTL = 60.0  # seconds


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                    follow_redirects=True,
                )
    return _client


def _encode_frame(frame: np.ndarray, quality: int = 85) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_crop(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


async def remote_detect(
    frame: np.ndarray,
    confidence_threshold: float = 0.5,
    target_classes: Optional[list[str]] = None,
) -> list[dict]:
    """Call remote ML server for YOLO object detection.

    Returns list of dicts with keys: class_name, confidence, bbox, crop (numpy).
    """
    client = await _get_client()
    payload = {
        "image": _encode_frame(frame),
        "confidence_threshold": confidence_threshold,
    }
    if target_classes:
        payload["target_classes"] = target_classes

    resp = await client.post(f"{ml_offload_url}/v1/detect", json=payload)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for det in data["detections"]:
        results.append({
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": tuple(det["bbox"]),
            "crop": _decode_crop(det["crop"]),
        })
    return results


async def remote_embedding(
    crop: np.ndarray,
    model: str = "cnn",
) -> Optional[list[float]]:
    """Call remote ML server for embedding computation (cnn or reid)."""
    client = await _get_client()
    payload = {
        "image": _encode_frame(crop),
        "model": model,
    }

    resp = await client.post(f"{ml_offload_url}/v1/embedding", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


async def remote_detect_faces(
    frame: np.ndarray,
    min_face_size: int = 20,
) -> list[dict]:
    """Call remote ML server for face detection + embedding.

    Returns list of dicts with keys: bbox, score, embedding, aligned_crop (numpy or None).
    """
    client = await _get_client()
    payload = {
        "image": _encode_frame(frame),
        "min_face_size": min_face_size,
    }

    resp = await client.post(f"{ml_offload_url}/v1/faces", json=payload)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for face in data["faces"]:
        aligned = None
        if face.get("aligned_crop"):
            aligned = _decode_crop(face["aligned_crop"])
        results.append({
            "bbox": tuple(face["bbox"]),
            "score": face["score"],
            "embedding": face.get("embedding"),
            "aligned_crop": aligned,
        })
    return results


async def health_check() -> dict:
    """Check remote ML server health."""
    try:
        client = await _get_client()
        resp = await client.get(f"{ml_offload_url}/health", timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def is_available() -> bool:
    """Check if the remote ML server is enabled and reachable (cached)."""
    global _health_ok, _health_checked_at

    if not ml_offload_enabled:
        return False

    now = time.monotonic()
    if now - _health_checked_at < _HEALTH_CACHE_TTL:
        return _health_ok

    result = await health_check()
    _health_ok = result.get("status") != "error"
    _health_checked_at = now

    if not _health_ok:
        logger.debug("ML server unavailable: %s", result.get("detail", "unknown"))
    return _health_ok


async def remote_describe(
    frame: np.ndarray,
    *,
    camera_name: str,
    object_type: str,
    named_object_name: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    instructions: Optional[str] = None,
) -> Optional[str]:
    """Send a snapshot to the remote ML server for LLM-based description.

    Returns a short description string, or None on any failure.
    """
    if not await is_available():
        return None

    try:
        # Down-scale large frames to reduce transfer size
        h, w = frame.shape[:2]
        _MAX_DIM = 512
        if max(h, w) > _MAX_DIM:
            scale = _MAX_DIM / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

        image_b64 = _encode_frame(frame, quality=80)

        ts = timestamp or datetime.now(timezone.utc)

        payload = {
            "image": image_b64,
            "camera_name": camera_name,
            "object_type": object_type,
            "timestamp": ts.isoformat(),
        }
        if named_object_name:
            payload["named_object_name"] = named_object_name
        if instructions:
            payload["instructions"] = instructions

        client = await _get_client()
        resp = await client.post(f"{ml_offload_url}/v1/describe", json=payload)
        resp.raise_for_status()
        data = resp.json()
        description = data.get("description", "").strip()

        if not description:
            return None

        logger.info(
            "ML narrative for %s @ %s: %s",
            named_object_name or object_type, camera_name, description[:120],
        )
        return description

    except httpx.TimeoutException:
        logger.debug("ML describe timeout for %s @ %s", object_type, camera_name)
        return None
    except Exception:
        logger.warning("ML describe failed", exc_info=True)
        return None


async def remote_vision_verify(
    detection_crop: np.ndarray,
    reference_crops: list[np.ndarray],
    *,
    candidate_name: str,
    object_type: str = "person",
    camera_name: str = "",
    candidate_attributes: dict | None = None,
) -> Optional[dict]:
    """Ask the remote vision model whether the detection matches the candidate.

    Returns dict with keys: match (bool), confidence (int 0-100), reasoning (str),
    inference_ms (float).  Returns None on any failure.
    """
    if not await is_available():
        return None
    if not reference_crops:
        return None

    try:
        det_b64 = _encode_frame(detection_crop, quality=85)
        ref_b64s = [_encode_frame(r, quality=85) for r in reference_crops[:3]]

        payload = {
            "detection_image": det_b64,
            "reference_images": ref_b64s,
            "candidate_name": candidate_name,
            "object_type": object_type,
            "camera_name": camera_name,
            "candidate_attributes": candidate_attributes or {},
        }

        client = await _get_client()
        resp = await client.post(
            f"{ml_offload_url}/v1/vision/verify",
            json=payload,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        resp.raise_for_status()
        data = resp.json()

        logger.info(
            "Vision verify %s vs '%s': match=%s conf=%d (%.0fms)",
            object_type, candidate_name, data.get("match"), data.get("confidence"),
            data.get("inference_ms", 0),
        )
        return data

    except httpx.TimeoutException:
        logger.debug("Vision verify timeout for %s vs %s", object_type, candidate_name)
        return None
    except Exception:
        logger.warning("Vision verify failed", exc_info=True)
        return None


async def remote_vision_identify(
    detection_crop: np.ndarray,
    candidates: list[dict],
    *,
    object_type: str = "person",
    camera_name: str = "",
) -> Optional[dict]:
    """Ask the remote vision model to identify the detection from a grid of candidates.

    candidates: list of dicts with keys 'name' (str) and 'crop' (np.ndarray).

    Returns dict with keys: identified_name (str|None), identified_index (int|None, 1-based),
    confidence (int 0-100), reasoning (str), inference_ms (float).  Returns None on failure.
    """
    if not await is_available():
        return None
    if not candidates:
        return None

    try:
        det_b64 = _encode_frame(detection_crop, quality=85)
        cand_payloads = [
            {
                "name": c["name"],
                "image": _encode_frame(c["crop"], quality=85),
                "attributes": c.get("attributes") or {},
            }
            for c in candidates[:8]
        ]

        payload = {
            "detection_image": det_b64,
            "candidates": cand_payloads,
            "object_type": object_type,
            "camera_name": camera_name,
        }

        client = await _get_client()
        resp = await client.post(
            f"{ml_offload_url}/v1/vision/identify",
            json=payload,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        resp.raise_for_status()
        data = resp.json()

        logger.info(
            "Vision identify %s: matched=%s conf=%d (%.0fms)",
            object_type, data.get("identified_name", "NONE"),
            data.get("confidence", 0), data.get("inference_ms", 0),
        )
        return data

    except httpx.TimeoutException:
        logger.debug("Vision identify timeout for %s", object_type)
        return None
    except Exception:
        logger.warning("Vision identify failed", exc_info=True)
        return None


async def remote_profile_analyze(
    images: list[np.ndarray],
    *,
    profile_name: str,
    object_type: str = "person",
) -> Optional[dict]:
    """Ask the remote vision model to check if all images show the same identity.

    Returns dict with keys: consistent (bool), outlier_indices (list[int] 0-based),
    confidence (int 0-100), reasoning (str), inference_ms (float).
    Returns None on failure.
    """
    if not await is_available():
        return None
    if len(images) < 2:
        return None

    try:
        image_payloads = [
            {"image": _encode_frame(img, quality=85), "label": ""}
            for img in images[:16]
        ]
        payload = {
            "profile_name": profile_name,
            "images": image_payloads,
            "object_type": object_type,
        }

        client = await _get_client()
        resp = await client.post(
            f"{ml_offload_url}/v1/vision/analyze-profile",
            json=payload,
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
        resp.raise_for_status()
        data = resp.json()

        logger.info(
            "Profile analyze '%s': consistent=%s outliers=%s conf=%d (%.0fms)",
            profile_name, data.get("consistent"), data.get("outlier_indices"),
            data.get("confidence", 0), data.get("inference_ms", 0),
        )
        return data

    except httpx.TimeoutException:
        logger.debug("Profile analyze timeout for %s", profile_name)
        return None
    except Exception:
        logger.warning("Profile analyze failed for %s", profile_name, exc_info=True)
        return None


async def remote_describe_activity(
    frames: list[np.ndarray],
    *,
    camera_name: str,
    object_type: str = "person",
    named_object_name: Optional[str] = None,
    duration_seconds: float = 0,
    previous_description: Optional[str] = None,
    instructions: Optional[str] = None,
) -> Optional[str]:
    """Send multiple chronological frames for an evolving activity description.

    Returns a description string, or None on failure.
    """
    if not await is_available():
        return None
    if not frames:
        return None

    try:
        # Down-scale frames
        _MAX_DIM = 512
        encoded = []
        for frame in frames[:6]:
            h, w = frame.shape[:2]
            if max(h, w) > _MAX_DIM:
                scale = _MAX_DIM / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            encoded.append(_encode_frame(frame, quality=80))

        payload = {
            "images": encoded,
            "camera_name": camera_name,
            "object_type": object_type,
            "duration_seconds": duration_seconds,
        }
        if named_object_name:
            payload["named_object_name"] = named_object_name
        if previous_description:
            payload["previous_description"] = previous_description
        if instructions:
            payload["instructions"] = instructions

        client = await _get_client()
        resp = await client.post(
            f"{ml_offload_url}/v1/describe/activity",
            json=payload,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        resp.raise_for_status()
        data = resp.json()
        desc = data.get("description", "").strip()

        if desc:
            logger.info("Activity narrative for %s @ %s: %s",
                        named_object_name or object_type, camera_name, desc[:120])
        return desc or None

    except httpx.TimeoutException:
        logger.debug("Activity describe timeout")
        return None
    except Exception:
        logger.warning("Activity describe failed", exc_info=True)
        return None
