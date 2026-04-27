"""BanusNas — Training API: named object management, training pipeline, and unrecognized detection review."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import logging
import subprocess

import cv2
import numpy as np
import asyncio
import json

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, or_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user
from core.config import settings
from models.database import async_session, get_session
from models.schemas import Camera, Event, EventType, NamedObject, ObjectCategory
from schemas.api_schemas import (
    CreateAndTrainRequest,
    NamedObjectCreate,
    NamedObjectResponse,
    NamedObjectUpdate,
    TrainFromEventsRequest,
)
from services.recognition_service import recognition_service, CNN_EMBED_DIM, REID_EMBED_DIM
from services.face_service import face_service, FACE_EMBED_DIM
from services.frigate_bridge import frigate_bridge
from services.attribute_estimator import get_display_attributes
from routers.search import _compute_retrain_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["training"], dependencies=[Depends(get_current_user)])


from sqlalchemy.orm.attributes import flag_modified as _flag_modified


async def _pin_event(ev: Event, reason: str = "manual", session: AsyncSession | None = None):
    """Mark an event as pinned with retention period from training settings."""
    from routers.system import load_training_settings
    async with async_session() as s:
        ts = await load_training_settings(s)
    pin_until = (datetime.now(timezone.utc) + timedelta(days=ts.training_retention_days)).isoformat()
    meta = dict(ev.metadata_extra or {})
    meta["pinned_until"] = pin_until
    meta["pinned_reason"] = reason
    ev.metadata_extra = meta
    _flag_modified(ev, "metadata_extra")


def _is_pinned(ev: Event) -> bool:
    """Check if an event is currently pinned (not expired)."""
    meta = ev.metadata_extra or {}
    pin_str = meta.get("pinned_until")
    if not pin_str:
        return False
    try:
        pin_dt = datetime.fromisoformat(pin_str)
        if pin_dt.tzinfo is None:
            pin_dt = pin_dt.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) < pin_dt
    except (ValueError, TypeError):
        return False


def _resolve_path(stored: str) -> Optional[Path]:
    """Resolve a stored file path, checking hot/cold fallback."""
    p = Path(stored)
    if p.exists():
        return p
    if settings.hot_storage_path:
        hot_snap = str(Path(settings.hot_storage_path) / "snapshots")
        cold_snap = settings.snapshots_path
        s = str(stored)
        if s.startswith(hot_snap):
            alt = Path(s.replace(hot_snap, cold_snap, 1))
        elif s.startswith(cold_snap):
            alt = Path(s.replace(cold_snap, hot_snap, 1))
        else:
            return None
        if alt.exists():
            return alt
    return None


def _load_event_crop(event) -> Optional[bytes]:
    """Load the best crop for an event — bbox crop from full snapshot, or thumbnail fallback.

    When bbox is available, crops from the full snapshot to isolate just the
    detected object.  This prevents training on background people/objects.
    Validates crop quality (blur/uniformity) to handle PTZ camera desync.
    """
    if event.bbox and event.snapshot_path:
        # Try bbox crop from full snapshot (clean version preferred)
        snap = _resolve_path(event.snapshot_path)
        if snap:
            # Prefer _clean.jpg (no annotations) over annotated snapshot
            clean = snap.parent / snap.name.replace(".jpg", "_clean.jpg")
            src = clean if clean.exists() else snap
            frame = cv2.imdecode(np.frombuffer(src.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                fh, fw = frame.shape[:2]
                try:
                    x1 = max(0, int(event.bbox.get("x1", 0)))
                    y1 = max(0, int(event.bbox.get("y1", 0)))
                    x2 = min(fw, int(event.bbox.get("x2", fw)))
                    y2 = min(fh, int(event.bbox.get("y2", fh)))
                except (TypeError, ValueError):
                    x1 = y1 = x2 = y2 = 0
                if x2 > x1 + 30 and y2 > y1 + 30:
                    bw, bh = x2 - x1, y2 - y1
                    pad_x, pad_y = int(bw * 0.15), int(bh * 0.15)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(fw, x2 + pad_x), min(fh, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    # Validate crop quality — reject blurry/empty crops (PTZ desync)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    pixel_std = float(np.std(gray))
                    if lap_var >= 25.0 and pixel_std >= 12.0:
                        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        if ok:
                            return buf.tobytes()

    # Fallback: use thumbnail (future thumbnails are already bbox crops)
    if event.thumbnail_path:
        thumb = _resolve_path(event.thumbnail_path)
        if thumb:
            return thumb.read_bytes()
    return None


# ──────────────── Crop Quality Gate ──────────────────
MIN_CROP_AREA = {"person": 1800, "pet": 800, "vehicle": 600, "other": 600}
MIN_BLUR_SCORE = 12.0  # Laplacian variance; lower = blurrier
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 6.0


def _assess_crop_quality(crop: np.ndarray, category: str) -> tuple[bool, str]:
    """Validate crop quality before using it for training.

    Returns (ok, reason). If ok is False, the crop should be skipped.
    """
    h, w = crop.shape[:2]
    area = h * w
    cat = category if category in MIN_CROP_AREA else "other"
    min_area = MIN_CROP_AREA[cat]
    if area < min_area:
        return False, f"too small ({area}px² < {min_area}px²)"
    aspect = w / h if h > 0 else 0
    if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
        return False, f"bad aspect ratio ({aspect:.2f})"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur < MIN_BLUR_SCORE:
        return False, f"too blurry (score {blur:.1f} < {MIN_BLUR_SCORE})"
    return True, "ok"


# ──────────────── Training Limits & Coverage ──────────────────
# Beyond these counts, running-average embeddings don't meaningfully improve.
MAX_TRAINING_IMAGES = {"person": 100, "pet": 40, "vehicle": 30, "other": 30}
TARGET_TRAINING_IMAGES = {"person": 60, "pet": 25, "vehicle": 20, "other": 20}


async def _detect_face_robust(
    crop: np.ndarray,
    snapshot_path: Optional[str] = None,
    bbox: Optional[dict] = None,
) -> tuple[list, np.ndarray]:
    """Try to detect a face using multiple fallback strategies.

    1. Detect directly on crop (thumbnail)
    2. Upscale small crops and retry
    3. Bbox-crop from full-res snapshot and retry
    4. Full snapshot as last resort

    Uses detect_faces_async which routes to the remote ML server when
    offloading is enabled (InsightFace not installed locally).

    Returns (faces_list, best_crop).
    """
    faces = await face_service.detect_faces_async(crop)
    if faces:
        return faces, crop

    # Upscale retry for small thumbnails
    h, w = crop.shape[:2]
    if h < 200 or w < 120:
        scale = min(2.5, max(200 / h, 120 / w))
        upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        faces = await face_service.detect_faces_async(upscaled)
        if faces:
            return faces, upscaled

    # Try full-res snapshot with bbox crop
    # Prefer the clean (unannotated) snapshot — annotated ones have bbox overlays
    # that corrupt face detection
    if snapshot_path:
        snap = _resolve_path(snapshot_path)
        if snap is None:
            snap = Path(snapshot_path)
        # Derive clean snapshot path: somefile.jpg -> somefile_clean.jpg
        clean_snap = snap.parent / snap.name.replace(".jpg", "_clean.jpg")
        snap_to_use = clean_snap if clean_snap.exists() else snap
        if snap_to_use.exists():
            full = cv2.imdecode(np.frombuffer(snap_to_use.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if full is not None:
                if bbox:
                    fh, fw = full.shape[:2]
                    bx1 = max(0, int(bbox.get("x1", bbox.get("0", 0))))
                    by1 = max(0, int(bbox.get("y1", bbox.get("1", 0))))
                    bx2 = min(fw, int(bbox.get("x2", bbox.get("2", fw))))
                    by2 = min(fh, int(bbox.get("y2", bbox.get("3", fh))))
                    if bx2 > bx1 and by2 > by1:
                        hi_crop = full[by1:by2, bx1:bx2]
                        faces = await face_service.detect_faces_async(hi_crop)
                        if faces:
                            return faces, hi_crop
                # Last resort: full frame
                faces = await face_service.detect_faces_async(full)
                if faces:
                    return faces, full

    return [], crop


def _get_face_embedding(face, crop: np.ndarray) -> list[float]:
    """Extract embedding from a FaceResult — works with both remote and local.

    Remote ML results already contain the embedding directly.
    Local InsightFace results need compute_face_embedding().
    """
    if face.embedding:
        return face.embedding
    if face.face_data is not None:
        emb, _ = face_service.compute_face_embedding(crop, face.face_data)
        return emb
    return []


def _compute_coverage(detections: list[dict], category: str) -> dict:
    """Analyse training coverage from detection metadata.

    Returns a dict describing model strengths, weaknesses, and suggested actions.
    Each detection dict must have: face_similarity, body_similarity, camera_name, has_face.
    """
    cat = category if category in MAX_TRAINING_IMAGES else "other"
    total = len(detections)
    max_imgs = MAX_TRAINING_IMAGES[cat]
    target = TARGET_TRAINING_IMAGES[cat]

    face_count = sum(1 for d in detections if d.get("has_face"))
    body_only = sum(1 for d in detections if not d.get("has_face"))
    cameras = list({d.get("camera_name", "Unknown") for d in detections})

    # Coverage scores (0-100)
    if cat == "person":
        face_score = min(face_count / 15, 1.0) * 100
        body_score = min(body_only / 8, 1.0) * 100
        cam_score = min(len(cameras) / 3, 1.0) * 100
        overall = (face_score * 0.45 + body_score * 0.30 + cam_score * 0.25)
    else:
        # Pets/vehicles/other: just need variety from multiple angles
        variety_score = min(total / target, 1.0) * 100
        cam_score = min(len(cameras) / 3, 1.0) * 100
        overall = (variety_score * 0.70 + cam_score * 0.30)
        face_score = 0
        body_score = 0

    # Build weakness messages
    tips: list[str] = []
    if cat == "person":
        if face_count < 5:
            tips.append("Needs more front-facing photos where the face is clearly visible")
        elif face_count < 15:
            tips.append(f"More face photos would help ({face_count}/15 ideal)")
        if body_only < 3:
            tips.append("Needs photos showing the back or side (no face visible) for body matching")
        elif body_only < 8:
            tips.append(f"More back/side views would improve body recognition ({body_only}/8 ideal)")
    elif cat == "pet":
        if total < 10:
            tips.append("Needs more photos from different angles — front, side, and back")
        elif total < target:
            tips.append(f"More varied photos will improve accuracy ({total}/{target} ideal)")
    else:
        if total < 10:
            tips.append("Needs more photos from different angles and distances")

    if len(cameras) < 2:
        tips.append("Try to include photos from different cameras for varied lighting")
    elif len(cameras) < 3:
        tips.append(f"Photos from more cameras help ({len(cameras)}/3+ ideal)")

    if total >= target and not tips:
        tips.append("Model has good coverage — ready for accurate detection")

    status = "excellent" if overall >= 80 else "good" if overall >= 55 else "needs_work" if overall >= 30 else "poor"

    return {
        "total": total,
        "max_images": max_imgs,
        "target_images": target,
        "face_count": face_count,
        "body_only_count": body_only,
        "camera_count": len(cameras),
        "cameras": cameras,
        "overall_score": round(overall),
        "face_score": round(face_score),
        "body_score": round(body_score) if cat == "person" else None,
        "camera_score": round(cam_score),
        "status": status,
        "tips": tips,
    }


# ──────────────────────── Named Objects CRUD ────────────────────────


@router.get("/objects", response_model=list[NamedObjectResponse])
async def list_named_objects(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(NamedObject).order_by(NamedObject.name))
    objects = result.scalars().all()

    responses = []
    for obj in objects:
        resp = NamedObjectResponse.model_validate(obj)

        # Get last seen info
        last_event = await session.execute(
            select(Event)
            .where(Event.named_object_id == obj.id)
            .order_by(desc(Event.started_at))
            .limit(1)
        )
        event = last_event.scalar_one_or_none()
        if event:
            resp.last_seen = event.started_at
            cam = await session.execute(select(Camera.name).where(Camera.id == event.camera_id))
            resp.last_camera = cam.scalar_one_or_none()

        responses.append(resp)
    return responses


@router.get("/objects/{object_id}", response_model=NamedObjectResponse)
async def get_named_object(object_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")
    return NamedObjectResponse.model_validate(obj)


@router.post("/objects", response_model=NamedObjectResponse, status_code=status.HTTP_201_CREATED)
async def create_named_object(data: NamedObjectCreate, session: AsyncSession = Depends(get_session)):
    obj = NamedObject(
        name=data.name,
        category=ObjectCategory(data.category),
    )
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return NamedObjectResponse.model_validate(obj)


@router.patch("/objects/{object_id}", response_model=NamedObjectResponse)
async def update_named_object(
    object_id: int,
    data: NamedObjectUpdate,
    session: AsyncSession = Depends(get_session),
):
    """Update a named object's name and/or manual attributes (gender, age_group)."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    if data.name is not None:
        obj.name = data.name

    # Manual gender / age_group overrides — stored with _manual flag
    attrs = dict(obj.attributes) if obj.attributes else {}
    if data.gender is not None:
        attrs["gender"] = data.gender
        attrs["gender_confidence"] = 1.0
        attrs["gender_samples"] = max(attrs.get("gender_samples", 0), 1)
        attrs["_gender_manual"] = True
        attrs["_gender_votes"] = {data.gender: 100.0}
    if data.age_group is not None:
        attrs["age_group"] = data.age_group
        attrs["age_group_confidence"] = 1.0
        attrs["age_group_samples"] = max(attrs.get("age_group_samples", 0), 1)
        attrs["_age_group_manual"] = True
        attrs["_age_group_votes"] = {data.age_group: 100.0}
    # Pet attributes — breed, color, markings
    if data.breed is not None:
        attrs["breed"] = data.breed
        attrs["_breed_manual"] = True
    if data.color is not None:
        attrs["color"] = data.color
        attrs["_color_manual"] = True
    if data.markings is not None:
        attrs["markings"] = data.markings
        attrs["_markings_manual"] = True
    # Vehicle attributes — vehicle_type, make
    if data.vehicle_type is not None:
        attrs["vehicle_type"] = data.vehicle_type
        attrs["_vehicle_type_manual"] = True
    if data.make is not None:
        attrs["make"] = data.make
        attrs["_make_manual"] = True
    obj.attributes = attrs

    await session.commit()
    await session.refresh(obj)
    frigate_bridge.invalidate_embedding_cache()
    return NamedObjectResponse.model_validate(obj)


@router.delete("/objects/{object_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_named_object(object_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    await session.delete(obj)
    await session.commit()


# ──────────────────── Training: Upload Images ──────────────────────


@router.post("/objects/{object_id}/train")
async def upload_training_images(
    object_id: int,
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session),
):
    """Upload reference images for training a named object."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    trained = 0
    skipped_quality = 0
    body_only_count = 0
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            continue
        img_bytes = await file.read()
        if len(img_bytes) > 10 * 1024 * 1024:  # 10MB limit
            continue

        result = await _train_object_with_bytes(obj, img_bytes)
        if result["ok"]:
            trained += 1
            obj.reference_image_count += 1
            if result["body_only"]:
                body_only_count += 1
        elif result["skip_reason"]:
            skipped_quality += 1

    session.add(obj)
    await session.commit()

    frigate_bridge.invalidate_embedding_cache()
    resp = {"message": f"{trained} training images added", "total_images": obj.reference_image_count}
    if skipped_quality:
        resp["skipped_quality"] = skipped_quality
    if body_only_count:
        resp["body_only_count"] = body_only_count
        resp["warning"] = f"{body_only_count} image(s) had no detectable face — trained body model only"
    return resp


# ──────────────────── Training: From Events ──────────────────────


@router.post("/objects/{object_id}/train-from-event")
async def train_from_event(
    object_id: int,
    event_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Use a detection event's crop as training data for a named object."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    event_result = await session.execute(select(Event).where(Event.id == event_id))
    event = event_result.scalar_one_or_none()
    if not event or not event.thumbnail_path:
        raise HTTPException(status_code=404, detail="Event or thumbnail not found")

    img_bytes = _load_event_crop(event)
    if not img_bytes:
        raise HTTPException(status_code=404, detail="No image available for training")

    result = await _train_object_with_bytes(obj, img_bytes)

    if result["ok"]:
        obj.reference_image_count += 1
        event.named_object_id = obj.id
        event.event_type = EventType.object_recognized
        session.add(obj)
        session.add(event)
        await session.commit()
        frigate_bridge.invalidate_embedding_cache()
        resp = {"message": "Training image added from event", "total_images": obj.reference_image_count}
        if result["body_only"]:
            resp["warning"] = "No face detected — trained body model only"
        return resp

    detail = result.get("skip_reason") or "Failed to train from event"
    raise HTTPException(status_code=400, detail=detail)


@router.post("/objects/{object_id}/train-from-events")
async def train_from_events(
    object_id: int,
    data: TrainFromEventsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Bulk-train a named object from multiple detection events."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    trained = 0
    for event_id in data.event_ids:
        ev = (await session.execute(select(Event).where(Event.id == event_id))).scalar_one_or_none()
        if not ev or not ev.thumbnail_path:
            continue

        img_bytes = _load_event_crop(ev)
        if not img_bytes:
            continue

        result = await _train_object_with_bytes(obj, img_bytes)
        if result["ok"]:
            trained += 1
            obj.reference_image_count += 1
            ev.named_object_id = obj.id
            ev.event_type = EventType.object_recognized
            await _pin_event(ev, reason="manual")
            session.add(ev)

    session.add(obj)
    await session.commit()
    frigate_bridge.invalidate_embedding_cache()
    return {"message": f"{trained} training images added", "total_images": obj.reference_image_count}


# ──────────────── Create + Train in One Step ──────────────────────


@router.post("/create-and-train", status_code=status.HTTP_201_CREATED)
async def create_and_train(
    data: CreateAndTrainRequest,
    session: AsyncSession = Depends(get_session),
):
    """Create a named object and immediately train from selected detection events."""
    obj = NamedObject(
        name=data.name,
        category=ObjectCategory(data.category),
    )
    session.add(obj)
    await session.flush()

    trained = 0
    for event_id in data.event_ids:
        ev = (await session.execute(select(Event).where(Event.id == event_id))).scalar_one_or_none()
        if not ev or not ev.thumbnail_path:
            continue

        img_bytes = _load_event_crop(ev)
        if not img_bytes:
            continue

        result = await _train_object_with_bytes(obj, img_bytes)
        if result["ok"]:
            trained += 1
            obj.reference_image_count += 1
            ev.named_object_id = obj.id
            ev.event_type = EventType.object_recognized
            await _pin_event(ev, reason="manual")
            session.add(ev)

    session.add(obj)
    await session.commit()
    await session.refresh(obj)

    return {"id": obj.id, "name": obj.name, "trained": trained, "total_images": obj.reference_image_count}


# ──────────────── Create + Train from Base64 Image ──────────────────────


@router.post("/create-and-train-image", status_code=status.HTTP_201_CREATED)
async def create_and_train_image(
    data: dict,
    session: AsyncSession = Depends(get_session),
):
    """Create or update a named object and train from a base64-encoded cropped image.

    Body: { image_b64, name?, category?, object_id? }
    If object_id is provided, adds training image to existing object.
    Otherwise creates a new named object with name+category.
    """
    import base64

    image_b64 = data.get("image_b64", "")
    object_id = data.get("object_id")

    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 is required")

    img_bytes = base64.b64decode(image_b64)

    if object_id:
        # Train existing object
        result = await session.execute(
            select(NamedObject).where(NamedObject.id == int(object_id))
        )
        obj = result.scalar_one_or_none()
        if not obj:
            raise HTTPException(status_code=404, detail="Named object not found")
    else:
        # Create new object
        name = data.get("name", "").strip()
        category = data.get("category", "other")
        if not name:
            raise HTTPException(status_code=400, detail="name is required for new objects")
        obj = NamedObject(
            name=name,
            category=ObjectCategory(category),
        )
        session.add(obj)
        await session.flush()

    result = await _train_object_with_bytes(obj, img_bytes)
    if result["ok"]:
        obj.reference_image_count += 1
    session.add(obj)
    await session.commit()
    await session.refresh(obj)

    resp = {"id": obj.id, "name": obj.name, "trained": 1 if result["ok"] else 0, "total_images": obj.reference_image_count}
    if result.get("body_only"):
        resp["warning"] = "No face detected — trained body model only"
    if result.get("skip_reason"):
        resp["skip_reason"] = result["skip_reason"]
    return resp


# ──────────────── Unrecognized Detections ──────────────────────


@router.get("/unrecognized")
async def get_unrecognized_detections(
    object_type: Optional[str] = Query(None, description="Filter: person, pet, or None for all"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Get recent unrecognized detections for review and training."""
    VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "bicycle", "boat", "train", "airplane"]
    PET_TYPES = ["cat", "dog"]

    filters = [
        Event.named_object_id.is_(None),
        Event.thumbnail_path.isnot(None),
        Event.event_type != EventType.motion,
    ]

    if object_type:
        if object_type == "pet":
            filters.append(Event.object_type.in_(PET_TYPES))
        elif object_type == "vehicle":
            filters.append(Event.object_type.in_(VEHICLE_TYPES))
        elif object_type == "other":
            filters.append(~Event.object_type.in_(["person"] + PET_TYPES + VEHICLE_TYPES))
        else:
            filters.append(Event.object_type == object_type)

    total = await session.scalar(
        select(func.count()).select_from(Event).where(and_(*filters))
    ) or 0

    result = await session.execute(
        select(Event)
        .where(and_(*filters))
        .order_by(desc(Event.started_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    events = result.scalars().all()

    items = []
    for ev in events:
        cam = await session.execute(select(Camera.name).where(Camera.id == ev.camera_id))
        cam_name = cam.scalar_one_or_none() or "Unknown"

        items.append({
            "event_id": ev.id,
            "camera_id": ev.camera_id,
            "camera_name": cam_name,
            "object_type": ev.object_type,
            "confidence": ev.confidence,
            "thumbnail_url": f"/api/events/{ev.id}/crop",
            "snapshot_url": f"/api/events/{ev.id}/snapshot" if ev.snapshot_path else None,
            "timestamp": ev.started_at.isoformat(),
        })

    return {"items": items, "total": total, "page": page, "page_size": page_size}


# ──────────────── Object Detections ──────────────────────


@router.get("/objects/{object_id}/detections")
async def get_object_detections(
    object_id: int,
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """Get recent detections for a named object."""
    result = await session.execute(
        select(Event)
        .where(Event.named_object_id == object_id)
        .order_by(desc(Event.started_at))
        .limit(limit)
    )
    events = result.scalars().all()
    return [
        {
            "event_id": e.id,
            "camera_id": e.camera_id,
            "object_type": e.object_type,
            "confidence": e.confidence,
            "timestamp": e.started_at.isoformat(),
            "thumbnail_url": f"/api/events/{e.id}/crop" if e.thumbnail_path else None,
        }
        for e in events
    ]


@router.get("/objects/{object_id}/profile")
async def get_object_profile(
    object_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a detailed profile for a named object — fast, no LLM calls."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    # Total detection count
    total = await session.scalar(
        select(func.count()).select_from(Event).where(Event.named_object_id == object_id)
    ) or 0

    # Camera breakdown — single join query instead of N+1
    cam_rows = (
        await session.execute(
            select(Event.camera_id, Camera.name, func.count().label("count"))
            .join(Camera, Camera.id == Event.camera_id)
            .where(Event.named_object_id == object_id)
            .group_by(Event.camera_id, Camera.name)
            .order_by(desc("count"))
        )
    ).all()

    cameras = [
        {"camera_id": cam_id, "camera_name": cam_name or "Unknown", "count": count}
        for cam_id, cam_name, count in cam_rows
    ]

    # Batch-load all camera names for detection mapping
    all_cam_rows = (await session.execute(select(Camera.id, Camera.name))).all()
    cam_name_map = {cid: cname for cid, cname in all_cam_rows}

    # Recent detections (last 50)
    det_result = await session.execute(
        select(Event)
        .where(Event.named_object_id == object_id)
        .order_by(desc(Event.started_at))
        .limit(50)
    )
    events = det_result.scalars().all()

    recent = [
        {
            "event_id": ev.id,
            "camera_id": ev.camera_id,
            "camera_name": cam_name_map.get(ev.camera_id, "Unknown"),
            "object_type": ev.object_type,
            "confidence": ev.confidence,
            "timestamp": ev.started_at.isoformat(),
            "thumbnail_url": f"/api/events/{ev.id}/crop" if ev.thumbnail_path else None,
            "snapshot_url": f"/api/events/{ev.id}/snapshot" if ev.snapshot_path else None,
            "narrative": (ev.metadata_extra or {}).get("narrative"),
        }
        for ev in events
    ]

    # Last seen info
    last_event = events[0] if events else None

    # Latest per-detection attributes (clothing, posture) from metadata_extra
    last_detection_attrs = None
    for ev in events:
        meta = ev.metadata_extra or {}
        if "attributes" in meta:
            last_detection_attrs = meta["attributes"]
            break

    # Profile image — use crop to show just the person/pet
    profile_image_url = None
    if last_event and last_event.thumbnail_path:
        profile_image_url = f"/api/events/{last_event.id}/crop"
    elif last_event and last_event.snapshot_path:
        profile_image_url = f"/api/events/{last_event.id}/crop"

    # Live tracking status (via Frigate bridge presence)
    is_live = False
    live_camera_id = None
    live_camera_name = None
    presence = frigate_bridge.get_current_presence()
    if obj.name in presence:
        is_live = True
        live_camera_id = presence[obj.name]
        # Look up camera name
        cam_result = await session.execute(select(Camera).where(Camera.id == live_camera_id))
        live_cam = cam_result.scalar_one_or_none()
        live_camera_name = live_cam.name if live_cam else None

    obj_data = {
        "id": obj.id,
        "name": obj.name,
        "category": obj.category.value if hasattr(obj.category, "value") else obj.category,
        "reference_image_count": obj.reference_image_count,
        "attributes": obj.attributes,
        "created_at": obj.created_at.isoformat(),
        "last_seen": last_event.started_at.isoformat() if last_event else None,
        "last_camera": cameras[0]["camera_name"] if cameras else None,
    }

    needs_retrain, retrain_reasons = _compute_retrain_status(obj, total)

    return {
        "object": obj_data,
        "total_detections": total,
        "cameras": cameras,
        "recent_detections": recent,
        "ai_summary": None,
        "profile_image_url": profile_image_url,
        "is_live": is_live,
        "live_camera_id": live_camera_id,
        "live_camera_name": live_camera_name,
        "attributes": get_display_attributes(obj.attributes),
        "last_detection_attrs": last_detection_attrs,
        "needs_retrain": needs_retrain,
        "retrain_reasons": retrain_reasons,
    }


@router.get("/objects/{object_id}/ai-summary")
async def get_object_ai_summary(
    object_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Generate AI summary separately — called lazily after profile loads."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    total = await session.scalar(
        select(func.count()).select_from(Event).where(Event.named_object_id == object_id)
    ) or 0

    cam_rows = (
        await session.execute(
            select(Event.camera_id, Camera.name, func.count().label("count"))
            .join(Camera, Camera.id == Event.camera_id)
            .where(Event.named_object_id == object_id)
            .group_by(Event.camera_id, Camera.name)
            .order_by(desc("count"))
        )
    ).all()
    cameras = [
        {"camera_id": cid, "camera_name": cn or "Unknown", "count": ct}
        for cid, cn, ct in cam_rows
    ]

    det_result = await session.execute(
        select(Event)
        .where(Event.named_object_id == object_id)
        .order_by(desc(Event.started_at))
        .limit(10)
    )
    events = det_result.scalars().all()
    cam_name_map = {cid: cn for cid, cn in (await session.execute(select(Camera.id, Camera.name))).all()}
    recent = [
        {
            "camera_name": cam_name_map.get(ev.camera_id, "Unknown"),
            "timestamp": ev.started_at.isoformat(),
        }
        for ev in events
    ]

    latest_snapshot_path = None
    for ev in events:
        if ev.snapshot_path:
            latest_snapshot_path = ev.snapshot_path
            break

    ai_summary = await _generate_activity_summary(
        obj.name,
        obj.category.value if hasattr(obj.category, "value") else obj.category,
        cameras, recent, total, latest_snapshot_path,
    )
    return {"ai_summary": ai_summary}


@router.post("/objects/{object_id}/rescan")
async def rescan_recent_recordings(
    object_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Rescan last 12 hours, return candidates with similarity scores for user confirmation."""
    from datetime import timedelta, timezone

    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    if not obj.embedding:
        raise HTTPException(status_code=400, detail="Object has no trained embedding")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=12)

    ev_result = await session.execute(
        select(Event)
        .where(and_(
            Event.started_at >= cutoff,
            Event.named_object_id.is_(None),
            Event.thumbnail_path.isnot(None),
            Event.event_type != EventType.motion,
        ))
        .order_by(desc(Event.started_at))
    )
    events = ev_result.scalars().all()

    stored_array = np.array(obj.embedding, dtype=np.float32)
    stored_norm = float(np.linalg.norm(stored_array))
    is_face_embedding = obj.category == ObjectCategory.person and len(obj.embedding) == FACE_EMBED_DIM
    match_threshold = 0.35 if is_face_embedding else 0.70

    candidates = []
    for ev in events:
        # Use bbox crop to isolate the target object
        img_bytes = _load_event_crop(ev)
        if not img_bytes:
            continue
        nparr = np.frombuffer(img_bytes, np.uint8)
        crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if crop is None:
            continue

        # Use face embedding for persons, CNN for others
        if is_face_embedding and face_service.is_available:
            # Only match person-type events
            if ev.object_type != "person":
                continue
            faces, crop = await _detect_face_robust(crop, ev.snapshot_path, ev.bbox)
            if not faces:
                continue
            emb = _get_face_embedding(faces[0], crop)
            if not emb:
                continue
            similarity = face_service.cosine_similarity(emb, obj.embedding)
        else:
            emb = recognition_service._compute_embedding_best(crop)
            emb_array = np.array(emb, dtype=np.float32)
            if len(emb_array) != len(stored_array):
                continue
            similarity = float(np.dot(emb_array, stored_array)) / max(stored_norm * float(np.linalg.norm(emb_array)), 1e-8)

        if similarity >= match_threshold:
            cam_name = (await session.scalar(select(Camera.name).where(Camera.id == ev.camera_id))) or "Unknown"
            candidates.append({
                "event_id": ev.id,
                "camera_name": cam_name,
                "object_type": ev.object_type,
                "similarity": round(similarity, 4),
                "thumbnail_url": f"/api/events/{ev.id}/crop",
                "snapshot_url": f"/api/events/{ev.id}/snapshot" if ev.snapshot_path else None,
                "timestamp": ev.started_at.isoformat(),
            })

    candidates.sort(key=lambda c: c["similarity"], reverse=True)

    return {"candidates": candidates, "scanned": len(events)}


@router.post("/objects/{object_id}/confirm-matches")
async def confirm_rescan_matches(
    object_id: int,
    data: TrainFromEventsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Confirm selected rescan candidates — assigns them to the object and reinforces embedding."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    confirmed = 0
    for event_id in data.event_ids:
        ev = (await session.execute(select(Event).where(Event.id == event_id))).scalar_one_or_none()
        if not ev or ev.named_object_id is not None:
            continue

        ev.named_object_id = obj.id
        ev.event_type = EventType.object_recognized
        await _pin_event(ev, reason="manual")
        session.add(ev)
        img_bytes = _load_event_crop(ev)
        if img_bytes:
            nparr = np.frombuffer(img_bytes, np.uint8)
            crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if crop is not None:
                if obj.category == ObjectCategory.person and face_service.is_available:
                    faces = await face_service.detect_faces_async(crop)
                    if faces:
                        emb = _get_face_embedding(faces[0], crop)
                        if emb:
                            obj.embedding = face_service.merge_face_embeddings(
                                obj.embedding, emb, obj.reference_image_count
                            )
                            obj.reference_image_count += 1
                else:
                    obj.embedding = recognition_service.compute_and_merge_embedding(
                        crop, obj.embedding, obj.reference_image_count
                    )
                    obj.reference_image_count += 1

        confirmed += 1

    session.add(obj)
    await session.commit()
    return {"confirmed": confirmed}


@router.post("/objects/{object_id}/audit")
async def audit_object_training(
    object_id: int,
    session: AsyncSession = Depends(get_session),
):
    """AI audit: compare each training detection's embedding against the model average, flag outliers."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    if not obj.embedding:
        raise HTTPException(status_code=400, detail="Object has no trained embedding")

    # Get all events assigned to this object
    ev_result = await session.execute(
        select(Event)
        .where(and_(Event.named_object_id == object_id, Event.thumbnail_path.isnot(None)))
        .order_by(desc(Event.started_at))
        .limit(200)
    )
    events = ev_result.scalars().all()

    stored_array = np.array(obj.embedding, dtype=np.float32)
    stored_norm = float(np.linalg.norm(stored_array))
    is_face_embedding = obj.category == ObjectCategory.person and len(obj.embedding) == FACE_EMBED_DIM

    detections = []
    similarities = []
    for ev in events:
        # Use bbox crop to isolate the correct person
        img_bytes = _load_event_crop(ev)
        if not img_bytes:
            continue
        crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if crop is None:
            continue

        # Use face embedding for persons, CNN/histogram for others
        if is_face_embedding and face_service.is_available:
            faces, crop = await _detect_face_robust(crop, ev.snapshot_path, ev.bbox)
            if faces:
                emb = _get_face_embedding(faces[0], crop)
                if emb:
                    sim = face_service.cosine_similarity(emb, obj.embedding)
                else:
                    sim = 0.0
            else:
                sim = 0.0  # No face found after all retries
        else:
            emb = recognition_service._compute_embedding_best(crop)
            emb_array = np.array(emb, dtype=np.float32)
            if len(emb_array) != len(stored_array):
                continue
            sim = float(np.dot(emb_array, stored_array)) / max(stored_norm * float(np.linalg.norm(emb_array)), 1e-8)
        similarities.append(sim)

        cam_name = (await session.scalar(select(Camera.name).where(Camera.id == ev.camera_id))) or "Unknown"
        detections.append({
            "event_id": ev.id,
            "camera_name": cam_name,
            "object_type": ev.object_type,
            "similarity": round(sim, 4),
            "thumbnail_url": f"/api/events/{ev.id}/crop",
            "timestamp": ev.started_at.isoformat(),
        })

    if not similarities:
        return {"detections": [], "summary": "No detections with thumbnails found.", "mean_similarity": 0, "flagged_count": 0}

    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities)) if len(similarities) > 1 else 0.0
    threshold = max(mean_sim - 2 * std_sim, 0.65)

    flagged = 0
    for d in detections:
        d["flagged"] = d["similarity"] < threshold
        if d["flagged"]:
            flagged += 1

    detections.sort(key=lambda d: d["similarity"])

    summary_parts = [f"Analyzed {len(detections)} detections."]
    summary_parts.append(f"Average similarity to model: {mean_sim:.1%}.")
    if flagged > 0:
        summary_parts.append(f"{flagged} detection{'s' if flagged != 1 else ''} flagged as potential mismatches (below {threshold:.1%}).")
        summary_parts.append("Review flagged items and remove any that don't belong to improve recognition accuracy.")
    else:
        summary_parts.append("All detections look consistent — no mismatches found.")

    return {
        "detections": detections,
        "summary": " ".join(summary_parts),
        "mean_similarity": round(mean_sim, 4),
        "flagged_count": flagged,
        "threshold": round(threshold, 4),
    }


@router.post("/objects/{object_id}/remove-detections")
async def remove_detections_from_object(
    object_id: int,
    data: TrainFromEventsRequest,
    session: AsyncSession = Depends(get_session),
):
    """Remove specific detections from a named object and recompute the embedding from remaining data."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    # Unlink the specified events (skip pinned)
    removed = 0
    skipped_pinned = 0
    for event_id in data.event_ids:
        ev = (await session.execute(select(Event).where(Event.id == event_id))).scalar_one_or_none()
        if ev and ev.named_object_id == object_id:
            if _is_pinned(ev):
                skipped_pinned += 1
                continue
            ev.named_object_id = None
            ev.event_type = EventType.object_detected
            session.add(ev)
            removed += 1

    # Recompute embedding from remaining assigned events
    remaining_result = await session.execute(
        select(Event)
        .where(and_(Event.named_object_id == object_id, Event.thumbnail_path.isnot(None)))
        .order_by(Event.started_at)
        .limit(200)
    )
    remaining = remaining_result.scalars().all()

    new_embedding = None
    count = 0
    for ev in remaining:
        thumb_path = Path(ev.thumbnail_path)
        if not thumb_path.exists():
            continue
        img_bytes = thumb_path.read_bytes()
        nparr = np.frombuffer(img_bytes, np.uint8)
        crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if crop is None:
            continue
        if obj.category == ObjectCategory.person and face_service.is_available:
            faces = await face_service.detect_faces_async(crop)
            if faces:
                emb = _get_face_embedding(faces[0], crop)
                if emb:
                    new_embedding = face_service.merge_face_embeddings(new_embedding, emb, count)
                    count += 1
        else:
            new_embedding = await recognition_service.compute_and_merge_embedding_async(crop, new_embedding, count)
            count += 1

    obj.embedding = new_embedding
    obj.reference_image_count = count
    session.add(obj)
    await session.commit()

    return {"removed": removed, "remaining_images": count, "skipped_pinned": skipped_pinned}


# ──────────────── Helpers ──────────────────────


async def _generate_activity_summary(
    name: str,
    category: str,
    cameras: list[dict],
    recent: list[dict],
    total: int,
    latest_snapshot_path: str | None = None,
) -> str | None:
    """Generate a natural language summary of an object's recent activity using LLM + vision."""
    if total == 0:
        return None

    from datetime import datetime as dt, timezone
    from zoneinfo import ZoneInfo
    local_tz = ZoneInfo("Europe/London")

    # ---------- Build structured context for LLM ----------
    cam_names = [c['camera_name'] for c in cameras[:5]]
    cam_info = ", ".join(f"{c['camera_name']} ({c['count']}x)" for c in cameras[:5])

    last_time_str = ""
    last_camera = ""
    if recent:
        latest_ts = recent[0]["timestamp"]
        try:
            latest = dt.fromisoformat(latest_ts)
            latest_local = latest.replace(tzinfo=timezone.utc) if latest.tzinfo is None else latest
            latest_local = latest_local.astimezone(local_tz)
            now_local = dt.now(local_tz)
            diff = now_local - latest_local
            hours = int(diff.total_seconds() / 3600)

            time_str = latest_local.strftime("%H:%M")
            date_str = latest_local.strftime("%d %b")

            if hours < 1:
                last_time_str = f"at {time_str} today"
            elif hours < 24:
                last_time_str = f"{hours} hour{'s' if hours != 1 else ''} ago at {time_str}"
            else:
                days = hours // 24
                last_time_str = f"on {date_str} at {time_str} ({days} day{'s' if days != 1 else ''} ago)"

            last_camera = recent[0].get("camera_name", "")
        except (ValueError, TypeError):
            pass

    # Try vision analysis of latest snapshot
    activity_description = ""
    if latest_snapshot_path:
        activity_description = await _analyze_snapshot_for_activity(
            latest_snapshot_path, name, category, last_camera,
        ) or ""

    # Try to generate an LLM narrative
    llm_summary = await _call_profile_llm(name, category, total, cam_info, last_time_str, last_camera, activity_description)
    if llm_summary:
        return llm_summary

    # ---------- Fallback: template-based summary ----------
    parts = [f"{name} has been detected {total} time{'s' if total != 1 else ''}"]
    if cameras:
        if len(cameras) == 1:
            parts.append(f"on {cam_names[0]}")
        elif len(cameras) <= 3:
            parts.append(f"across {', '.join(cam_names)}")
        else:
            parts.append(f"across {', '.join(cam_names[:3])} and {len(cameras) - 3} other camera{'s' if len(cameras) - 3 > 1 else ''}")

    summary = " ".join(parts) + "."

    if last_time_str:
        summary += f" Last seen {last_time_str}."
        if last_camera:
            summary += f" Spotted on {last_camera}."

    if activity_description:
        summary += f" {activity_description}"

    if cameras and cameras[0]["count"] > total * 0.5 and len(cameras) > 1:
        summary += f" Most frequently appears on {cameras[0]['camera_name']}."

    return summary


async def _analyze_snapshot_for_activity(
    snapshot_path: str, name: str, category: str, camera_name: str,
) -> str | None:
    """Use Ollama vision model to describe what the subject is doing in the latest snapshot."""
    import base64
    try:
        import httpx

        img_path = Path(snapshot_path)
        if not img_path.exists():
            return None

        import asyncio
        img_bytes = await asyncio.to_thread(img_path.read_bytes)
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        entity_type = "person" if category == "person" else "animal" if category == "pet" else category
        prompt = (
            f"This is a CCTV camera image from the '{camera_name}' camera. "
            f"The detected subject is '{name}' (a {entity_type}). "
            "In one sentence, describe what they are currently doing and the scene around them. "
            "Be specific about their activity (e.g. sitting on the sofa, playing with toys, "
            "eating from a bowl, walking through the hallway). Don't describe the image quality."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_url}/api/chat",
                json={
                    "model": settings.ollama_vision_model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                    "stream": False,
                },
            )
            response.raise_for_status()
            desc = response.json().get("message", {}).get("content", "").strip()
            if desc:
                logger.info("Vision activity for %s: %s", name, desc[:120])
                return desc
    except Exception as e:
        logger.debug("Vision analysis skipped for %s: %s", name, e)
    return None


async def _call_profile_llm(
    name: str, category: str, total: int, cam_info: str,
    last_time_str: str, last_camera: str, activity_description: str,
) -> str | None:
    """Call Ollama text model to generate a natural profile summary."""
    try:
        import httpx

        entity_type = "person" if category == "person" else "pet" if category == "pet" else category

        user_prompt = (
            f"Write a 2-3 sentence activity summary for '{name}' (a {entity_type}) from a home CCTV system.\n"
            f"Total detections: {total}\n"
            f"Cameras spotted on (with counts): {cam_info}\n"
        )
        if last_time_str:
            user_prompt += f"Last seen: {last_time_str} on {last_camera}\n"
        if activity_description:
            user_prompt += f"Latest activity from photo analysis: {activity_description}\n"
        user_prompt += (
            "\nWrite a warm, concise summary. Include what they were last doing if a photo description "
            "is provided. Mention their favourite area if one camera dominates. Don't use bullet points."
        )

        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                f"{settings.ollama_url}/v1/chat/completions",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {"role": "system", "content": "You are a friendly smart home assistant writing brief activity profiles. Be warm and specific. No markdown."},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150,
                },
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"].strip()
            if text:
                logger.info("LLM profile summary for %s: %s", name, text[:120])
                return text
    except Exception as e:
        logger.debug("LLM profile summary skipped for %s: %s", name, e)
    return None


# ──────────────────── Deep Retrain Wizard API ────────────────────


@router.post("/objects/{object_id}/deep-retrain/existing")
async def deep_retrain_existing(
    object_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Step 1: Return all existing assigned detections with per-image similarity scores.

    Groups images into pages of 6 for user review. Each image gets a similarity
    score against the current model average so users can spot outliers.
    """
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    ev_result = await session.execute(
        select(Event)
        .where(and_(Event.named_object_id == object_id, Event.thumbnail_path.isnot(None)))
        .order_by(desc(Event.started_at))
        .limit(500)
    )
    events = ev_result.scalars().all()

    # Pre-fetch camera names in bulk (avoid N+1 queries)
    cam_ids = list({ev.camera_id for ev in events})
    if cam_ids:
        cam_rows = await session.execute(select(Camera.id, Camera.name).where(Camera.id.in_(cam_ids)))
        cam_map = {row.id: row.name for row in cam_rows}
    else:
        cam_map = {}

    stored_emb = np.array(obj.embedding, dtype=np.float32) if obj.embedding else None
    stored_body = np.array(obj.body_embedding, dtype=np.float32) if obj.body_embedding else None
    is_person = obj.category == ObjectCategory.person
    obj_embedding = obj.embedding  # snapshot for concurrent access

    # Prepare lightweight event data for parallel processing (no DB access needed)
    event_list = [
        {"id": ev.id, "camera_id": ev.camera_id, "confidence": ev.confidence,
         "snapshot_path": ev.snapshot_path, "bbox": ev.bbox,
         "started_at": ev.started_at.isoformat(), "thumbnail_path": ev.thumbnail_path}
        for ev in events
    ]

    CONCURRENCY = 5
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _process_existing(ed):
        async with sem:
            thumb_path = _resolve_path(ed["thumbnail_path"])
            if not thumb_path:
                return None
            img_bytes = await asyncio.to_thread(thumb_path.read_bytes)
            nparr = np.frombuffer(img_bytes, np.uint8)
            crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if crop is None:
                return None

            face_sim = None
            body_sim = None
            has_face = False

            if is_person and face_service.is_available:
                faces, crop = await _detect_face_robust(crop, ed.get("snapshot_path"), ed.get("bbox"))
                if faces:
                    has_face = True
                    if stored_emb is not None and len(stored_emb) == FACE_EMBED_DIM:
                        emb = _get_face_embedding(faces[0], crop)
                        if emb:
                            face_sim = round(float(face_service.cosine_similarity(emb, obj_embedding)), 4)
                if stored_body is not None and recognition_service.reid_available:
                    body_emb = await asyncio.to_thread(recognition_service.compute_reid_embedding, crop)
                    if body_emb:
                        b = np.array(body_emb, dtype=np.float32)
                        body_sim = round(float(np.dot(b, stored_body) / max(np.linalg.norm(b) * np.linalg.norm(stored_body), 1e-8)), 4)
            elif not is_person and stored_emb is not None:
                emb = await asyncio.to_thread(recognition_service._compute_embedding_best, crop)
                ea = np.array(emb, dtype=np.float32)
                if len(ea) == len(stored_emb):
                    face_sim = round(float(np.dot(ea, stored_emb) / max(np.linalg.norm(ea) * np.linalg.norm(stored_emb), 1e-8)), 4)

            best_sim = max(filter(None, [face_sim, body_sim]), default=0.0)
            if is_person and face_sim is not None and body_sim is not None:
                if face_sim >= 0.40:
                    best_sim = face_sim * 0.65 + body_sim * 0.35
                elif face_sim < 0.30:
                    best_sim = face_sim * 0.3 + body_sim * 0.15
                else:
                    best_sim = face_sim * 0.5 + body_sim * 0.5
            elif is_person and face_sim is None and body_sim is not None:
                best_sim = body_sim * 0.80

            cam_name = cam_map.get(ed["camera_id"], "Unknown")
            return {
                "event_id": ed["id"],
                "camera_name": cam_name,
                "confidence": ed["confidence"],
                "face_similarity": face_sim,
                "body_similarity": body_sim,
                "best_similarity": best_sim,
                "has_face": has_face,
                "thumbnail_url": f"/api/events/{ed['id']}/crop",
                "snapshot_url": f"/api/events/{ed['id']}/snapshot" if ed.get("snapshot_path") else None,
                "timestamp": ed["started_at"],
            }

    results = await asyncio.gather(*[_process_existing(ed) for ed in event_list])
    detections = [r for r in results if r is not None]

    detections.sort(key=lambda d: d["best_similarity"], reverse=True)

    cat_val = obj.category.value if hasattr(obj.category, "value") else obj.category
    coverage = _compute_coverage(detections, cat_val)

    return {
        "object_id": obj.id,
        "object_name": obj.name,
        "category": cat_val,
        "current_refs": obj.reference_image_count,
        "detections": detections,
        "total": len(detections),
        "coverage": coverage,
    }


@router.post("/objects/{object_id}/deep-retrain/scan")
async def deep_retrain_scan(
    object_id: int,
    hours: int = Query(24, ge=1, le=72),
    data: dict | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Step 2: Scan recent events for new matches using face + body ReID.

    Returns NDJSON stream with progress updates and final result.
    Face is weighted as the primary signal for persons — if a face is
    detected but doesn't match, body similarity is discounted.
    Includes events already system-matched to this object so user can verify.
    """
    from datetime import timedelta, timezone

    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    if not obj.embedding and not obj.body_embedding:
        raise HTTPException(status_code=400, detail="Object has no trained embedding — confirm existing detections first")

    # Event IDs already shown in step 1 — skip them
    exclude_ids = set(data.get("exclude_event_ids", [])) if data else set()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    is_person = obj.category == ObjectCategory.person

    type_filter = Event.object_type == "person" if is_person else Event.object_type != "person"

    # Include unrecognized events AND events already assigned to THIS object
    ev_result = await session.execute(
        select(Event)
        .where(and_(
            Event.started_at >= cutoff,
            or_(Event.named_object_id.is_(None), Event.named_object_id == obj.id),
            Event.thumbnail_path.isnot(None),
            Event.event_type != EventType.motion,
            type_filter,
        ))
        .order_by(desc(Event.started_at))
        .limit(1000)
    )
    events = ev_result.scalars().all()

    # Pre-fetch camera names in one query to avoid N+1 in the loop
    cam_ids = list({ev.camera_id for ev in events})
    cam_rows = await session.execute(select(Camera.id, Camera.name).where(Camera.id.in_(cam_ids))) if cam_ids else None
    cam_map = {row.id: row.name for row in cam_rows} if cam_rows else {}

    stored_face = np.array(obj.embedding, dtype=np.float32) if obj.embedding else None
    stored_body = np.array(obj.body_embedding, dtype=np.float32) if obj.body_embedding else None
    is_face_emb = is_person and stored_face is not None and len(stored_face) == FACE_EMBED_DIM
    obj_id = obj.id
    obj_name = obj.name
    obj_embedding = obj.embedding

    # Prepare lightweight event data to avoid DB access in thread
    event_data = []
    for ev in events:
        if ev.id in exclude_ids:
            continue
        thumb_path = ev.thumbnail_path
        if thumb_path:
            event_data.append({
                "id": ev.id,
                "camera_id": ev.camera_id,
                "confidence": ev.confidence,
                "snapshot_path": ev.snapshot_path,
                "bbox": ev.bbox,
                "started_at": ev.started_at.isoformat(),
                "thumbnail_path": thumb_path,
                "system_matched": ev.named_object_id == obj.id,
            })

    total = len(event_data)

    async def generate():
        # Send initial progress
        yield json.dumps({"type": "progress", "scanned": 0, "total": total, "found": 0}) + "\n"

        candidates = []

        async def process_one(ed):
            """Process a single event."""
            # System-matched events are always included — skip expensive ML
            if ed.get("system_matched"):
                return {
                    "event_id": ed["id"],
                    "camera_name": cam_map.get(ed["camera_id"], "Unknown"),
                    "confidence": ed["confidence"],
                    "face_similarity": None,
                    "body_similarity": None,
                    "best_similarity": 1.0,
                    "has_face": False,
                    "thumbnail_url": f"/api/events/{ed['id']}/crop",
                    "snapshot_url": f"/api/events/{ed['id']}/snapshot" if ed["snapshot_path"] else None,
                    "timestamp": ed["started_at"],
                    "system_matched": True,
                }

            path = _resolve_path(ed["thumbnail_path"])
            if not path:
                return None
            # Try bbox crop from snapshot for cleaner matching
            crop = None
            bbox = ed.get("bbox")
            snap_path = ed.get("snapshot_path")
            if bbox and snap_path:
                snap = _resolve_path(snap_path)
                if snap:
                    snap_bytes = await asyncio.to_thread(snap.read_bytes)
                    frame = cv2.imdecode(np.frombuffer(snap_bytes, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        fh, fw = frame.shape[:2]
                        try:
                            bx1 = max(0, int(bbox.get("x1", 0)))
                            by1 = max(0, int(bbox.get("y1", 0)))
                            bx2 = min(fw, int(bbox.get("x2", fw)))
                            by2 = min(fh, int(bbox.get("y2", fh)))
                        except (TypeError, ValueError):
                            bx1 = by1 = bx2 = by2 = 0
                        if bx2 > bx1 + 30 and by2 > by1 + 30:
                            bw, bh = bx2 - bx1, by2 - by1
                            px, py = int(bw * 0.15), int(bh * 0.15)
                            crop = frame[max(0,by1-py):min(fh,by2+py), max(0,bx1-px):min(fw,bx2+px)]
            if crop is None:
                img_bytes = await asyncio.to_thread(path.read_bytes)
                nparr = np.frombuffer(img_bytes, np.uint8)
                crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if crop is None:
                return None

            face_sim = None
            body_sim = None
            face_detected = False

            if is_person:
                # Body-first early rejection: cheaper than face detection
                if stored_body is not None and recognition_service.reid_available:
                    body_emb = await asyncio.to_thread(recognition_service.compute_reid_embedding, crop)
                    if body_emb:
                        b = np.array(body_emb, dtype=np.float32)
                        body_sim = float(np.dot(b, stored_body) / max(np.linalg.norm(b) * np.linalg.norm(stored_body), 1e-8))
                    # Skip expensive face detection if body clearly doesn't match
                    if body_sim is not None and body_sim < 0.20:
                        return None

                if face_service.is_available and is_face_emb:
                    # Lightweight face detection for scan: thumbnail + upscale only
                    # (skip expensive snapshot fallback — saves ~100ms/image)
                    faces = await face_service.detect_faces_async(crop)
                    if not faces:
                        h, w = crop.shape[:2]
                        if h < 200 or w < 120:
                            scale = min(2.5, max(200 / h, 120 / w))
                            upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                            faces = await face_service.detect_faces_async(upscaled)
                            if faces:
                                crop = upscaled
                    if faces:
                        face_detected = True
                        emb = _get_face_embedding(faces[0], crop)
                        if emb:
                            face_sim = float(face_service.cosine_similarity(emb, obj_embedding))
            else:
                if stored_face is not None:
                    emb = recognition_service._compute_embedding_best(crop)
                    ea = np.array(emb, dtype=np.float32)
                    if len(ea) == len(stored_face):
                        face_sim = float(np.dot(ea, stored_face) / max(np.linalg.norm(ea) * np.linalg.norm(stored_face), 1e-8))

            # Compute best_similarity: face-weighted for persons
            if is_person:
                if face_sim is not None and body_sim is not None:
                    if face_sim >= 0.40:
                        # Strong face match — face primary, body bonus
                        best = face_sim * 0.65 + body_sim * 0.35
                    elif face_detected and face_sim < 0.30:
                        # Face clearly doesn't match — discount body heavily
                        # Similar clothes on a different person
                        best = face_sim * 0.3 + body_sim * 0.15
                    else:
                        # Borderline face — balanced
                        best = face_sim * 0.5 + body_sim * 0.5
                elif face_sim is not None:
                    best = face_sim
                elif body_sim is not None:
                    # No face detected — body only, discounted
                    best = body_sim * 0.80
                else:
                    best = 0.0
            else:
                best = max(filter(None, [face_sim, body_sim]), default=0.0)

            # Inclusion thresholds (relaxed to catch more candidates for user review)
            include = False
            if is_person:
                if face_sim is not None and face_sim >= 0.35:
                    include = True
                elif face_sim is not None and face_sim >= 0.25 and body_sim is not None and body_sim >= 0.40:
                    include = True  # Borderline face + decent body
                elif face_sim is None and body_sim is not None and body_sim >= 0.45:
                    include = True  # No face detected, body match
                elif best >= 0.35:
                    include = True  # Catch-all for combined score
            else:
                include = best >= 0.45

            # System-matched events are always included (user needs to verify them)
            if include or ed.get("system_matched"):
                return {
                    "event_id": ed["id"],
                    "camera_name": cam_map.get(ed["camera_id"], "Unknown"),
                    "confidence": ed["confidence"],
                    "face_similarity": round(face_sim, 4) if face_sim else None,
                    "body_similarity": round(body_sim, 4) if body_sim else None,
                    "best_similarity": round(best, 4),
                    "has_face": face_detected,
                    "thumbnail_url": f"/api/events/{ed['id']}/crop",
                    "snapshot_url": f"/api/events/{ed['id']}/snapshot" if ed["snapshot_path"] else None,
                    "timestamp": ed["started_at"],
                    "system_matched": ed.get("system_matched", False),
                }
            return None

        SCAN_BATCH = 15
        for batch_start in range(0, total, SCAN_BATCH):
            batch = event_data[batch_start:batch_start + SCAN_BATCH]
            batch_results = await asyncio.gather(*[process_one(ed) for ed in batch])
            new_candidates = [r for r in batch_results if r is not None]
            candidates.extend(new_candidates)
            scanned = min(batch_start + len(batch), total)
            yield json.dumps({
                "type": "progress", "scanned": scanned, "total": total,
                "found": len(candidates), "new_candidates": new_candidates,
            }) + "\n"

        candidates.sort(key=lambda c: c["best_similarity"], reverse=True)
        yield json.dumps({
            "type": "done",
            "object_id": obj_id,
            "object_name": obj_name,
            "scanned": total,
            "found": len(candidates),
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/objects/{object_id}/deep-retrain/rescore")
async def deep_retrain_rescore(
    object_id: int,
    data: dict,
    session: AsyncSession = Depends(get_session),
):
    """Re-score unreviewed detections after user has confirmed/rejected some.

    Body: { "confirmed_ids": [int, ...], "rejected_ids": [int, ...], "unreviewed_ids": [int, ...] }

    Builds a temporary embedding from confirmed images AND a rejection
    embedding from rejected images. Candidates similar to rejections
    get penalized, making the ranking actually improve with feedback.
    """
    confirmed_ids = data.get("confirmed_ids", [])
    rejected_ids = data.get("rejected_ids", [])
    unreviewed_ids = data.get("unreviewed_ids", [])

    logger.info(f"Rescore object {object_id}: {len(confirmed_ids)} confirmed, {len(rejected_ids)} rejected, {len(unreviewed_ids)} unreviewed")

    if len(confirmed_ids) < 2:
        return {"rescored": []}

    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    is_person = obj.category == ObjectCategory.person

    async def _build_embedding(event_ids):
        """Build merged face + body embedding from a list of event IDs (parallel)."""
        if not event_ids:
            return None, None, 0, 0
        # Batch-load events in one query
        ev_result = await session.execute(select(Event).where(Event.id.in_(event_ids)))
        ev_map = {ev.id: ev for ev in ev_result.scalars().all()}

        sem_build = asyncio.Semaphore(5)

        async def _extract(eid):
            ev = ev_map.get(eid)
            if not ev or not ev.thumbnail_path:
                return None, None
            thumb = Path(ev.thumbnail_path)
            if not thumb.exists():
                return None, None
            async with sem_build:
                img_bytes = await asyncio.to_thread(thumb.read_bytes)
                crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if crop is None:
                    return None, None
                f_emb = None
                b_emb = None
                if is_person and face_service.is_available:
                    faces = await face_service.detect_faces_async(crop)
                    if faces:
                        f_emb = _get_face_embedding(faces[0], crop)
                    if recognition_service.reid_available:
                        b_emb = await asyncio.to_thread(recognition_service.compute_reid_embedding, crop)
                else:
                    f_emb = await asyncio.to_thread(recognition_service._compute_embedding_best, crop)
                return f_emb, b_emb

        results = await asyncio.gather(*[_extract(eid) for eid in event_ids])

        # Batch-accumulate embeddings (equal weighting, single normalization)
        face_sum = None
        body_sum = None
        fc = 0
        bc = 0
        for f_emb, b_emb in results:
            if f_emb is not None and f_emb:
                arr = np.array(f_emb, dtype=np.float64)
                face_sum = arr if face_sum is None else face_sum + arr
                fc += 1
            if b_emb is not None and b_emb:
                arr = np.array(b_emb, dtype=np.float64)
                body_sum = arr if body_sum is None else body_sum + arr
                bc += 1

        face_emb = None
        if face_sum is not None and fc > 0:
            avg = face_sum / fc
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            face_emb = avg.tolist()

        body_emb = None
        if body_sum is not None and bc > 0:
            avg = body_sum / bc
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            body_emb = avg.tolist()

        return face_emb, body_emb, fc, bc

    # Build confirmed embedding
    temp_face_emb, temp_body_emb, face_count, body_count = await _build_embedding(confirmed_ids)
    if not temp_face_emb and not temp_body_emb:
        return {"rescored": []}

    # Build rejection embedding (if any rejections)
    if rejected_ids:
        rej_face_emb, rej_body_emb, _, _ = await _build_embedding(rejected_ids)
    else:
        rej_face_emb, rej_body_emb = None, None

    stored_face = np.array(temp_face_emb, dtype=np.float32) if temp_face_emb else None
    stored_body = np.array(temp_body_emb, dtype=np.float32) if temp_body_emb else None
    rej_face = np.array(rej_face_emb, dtype=np.float32) if rej_face_emb else None
    rej_body = np.array(rej_body_emb, dtype=np.float32) if rej_body_emb else None
    is_face_emb = is_person and stored_face is not None and len(stored_face) == FACE_EMBED_DIM

    def _cosine(a, b):
        return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))

    # Batch-load unreviewed events in one query
    if unreviewed_ids:
        unrev_result = await session.execute(select(Event).where(Event.id.in_(unreviewed_ids)))
        unrev_map = {ev.id: ev for ev in unrev_result.scalars().all()}
    else:
        unrev_map = {}

    sem_rescore = asyncio.Semaphore(5)

    async def _rescore_one(event_id):
        ev = unrev_map.get(event_id)
        if not ev or not ev.thumbnail_path:
            return None
        thumb = Path(ev.thumbnail_path)
        if not thumb.exists():
            return None
        async with sem_rescore:
            img_bytes = await asyncio.to_thread(thumb.read_bytes)
            nparr = np.frombuffer(img_bytes, np.uint8)
            crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if crop is None:
                return None

            face_sim = None
            body_sim = None
            rej_penalty = 0.0

            if is_person:
                if is_face_emb and face_service.is_available:
                    faces = await face_service.detect_faces_async(crop)
                    if faces:
                        emb = _get_face_embedding(faces[0], crop)
                        if emb:
                            face_sim = _cosine(np.array(emb, dtype=np.float32), stored_face)
                            if rej_face is not None:
                                rej_sim = _cosine(np.array(emb, dtype=np.float32), rej_face)
                                if rej_sim > 0.25:
                                    rej_penalty = max(rej_penalty, rej_sim * 0.7)
                if stored_body is not None and recognition_service.reid_available:
                    b_emb = await asyncio.to_thread(recognition_service.compute_reid_embedding, crop)
                    if b_emb:
                        b = np.array(b_emb, dtype=np.float32)
                        body_sim = _cosine(b, stored_body)
                        if rej_body is not None:
                            rej_sim = _cosine(b, rej_body)
                            if rej_sim > 0.25:
                                rej_penalty = max(rej_penalty, rej_sim * 0.7)
            elif stored_face is not None:
                emb = await asyncio.to_thread(recognition_service._compute_embedding_best, crop)
                ea = np.array(emb, dtype=np.float32)
                if len(ea) == len(stored_face):
                    face_sim = _cosine(ea, stored_face)
                    if rej_face is not None and len(rej_face) == len(ea):
                        rej_sim = _cosine(ea, rej_face)
                        if rej_sim > 0.25:
                            rej_penalty = rej_sim * 0.7

            # Weighted scoring for persons (face primary)
            if is_person and face_sim is not None and body_sim is not None:
                if face_sim >= 0.40:
                    raw_best = face_sim * 0.65 + body_sim * 0.35
                elif face_sim < 0.30:
                    raw_best = face_sim * 0.3 + body_sim * 0.15
                else:
                    raw_best = face_sim * 0.5 + body_sim * 0.5
            elif is_person and face_sim is None and body_sim is not None:
                raw_best = body_sim * 0.80
            else:
                raw_best = max(filter(None, [face_sim, body_sim]), default=0.0)
            best = max(raw_best - rej_penalty, 0.0)

            return {
                "event_id": event_id,
                "face_similarity": round(face_sim, 4) if face_sim is not None else None,
                "body_similarity": round(body_sim, 4) if body_sim is not None else None,
                "best_similarity": round(best, 4),
            }

    rescore_results = await asyncio.gather(*[_rescore_one(eid) for eid in unreviewed_ids])
    rescored = [r for r in rescore_results if r is not None]

    rescored.sort(key=lambda r: r["best_similarity"], reverse=True)
    penalized = sum(1 for r in rescored if r["best_similarity"] < max(filter(None, [r.get("face_similarity"), r.get("body_similarity")]), default=0))
    logger.info(f"Rescore complete: {len(rescored)} candidates scored, {penalized} penalized by rejection similarity")
    return {"rescored": rescored, "confirmed_count": len(confirmed_ids), "rejected_count": len(rejected_ids), "face_embeddings": face_count, "body_embeddings": body_count}


@router.post("/objects/{object_id}/deep-retrain/commit")
async def deep_retrain_commit(
    object_id: int,
    data: dict,
    session: AsyncSession = Depends(get_session),
):
    """Step 3: Rebuild the model from only user-confirmed event IDs.

    Body: { "confirmed_event_ids": [int, ...], "new_event_ids": [int, ...] }
    - confirmed_event_ids: existing detections the user confirmed as correct
    - new_event_ids: new scan candidates the user accepted

    Discards old embedding entirely and recomputes from scratch using
    only trusted images for maximum accuracy.
    """
    confirmed_ids = data.get("confirmed_event_ids", [])
    new_ids = data.get("new_event_ids", [])
    all_ids = list(set(confirmed_ids + new_ids))

    if len(all_ids) < 3:
        raise HTTPException(status_code=400, detail="At least 3 confirmed images required for accurate retraining")

    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    is_person = obj.category == ObjectCategory.person
    cat = obj.category.value if hasattr(obj.category, "value") else obj.category
    max_imgs = MAX_TRAINING_IMAGES.get(cat, 30)

    # Cap at max — balanced selection for persons to ensure face/body mix
    if len(all_ids) > max_imgs:
        if is_person and face_service.is_available:
            logger.info(f"Commit: balanced capping {len(all_ids)} images to {max_imgs} (person)")
            # Pre-classify: parallel face detection pass
            cap_ev_result = await session.execute(select(Event).where(Event.id.in_(all_ids)))
            cap_ev_map = {ev.id: ev for ev in cap_ev_result.scalars().all()}
            sem_cap = asyncio.Semaphore(5)

            async def _classify(eid):
                ev = cap_ev_map.get(eid)
                if not ev or not ev.thumbnail_path:
                    return eid, None
                thumb = Path(ev.thumbnail_path)
                if not thumb.exists():
                    return eid, None
                async with sem_cap:
                    crop = cv2.imdecode(np.frombuffer(await asyncio.to_thread(thumb.read_bytes), np.uint8), cv2.IMREAD_COLOR)
                    if crop is None:
                        return eid, None
                    faces = await face_service.detect_faces_async(crop)
                    return eid, bool(faces)

            cap_results = await asyncio.gather(*[_classify(eid) for eid in all_ids])
            face_ids: list[int] = [eid for eid, has in cap_results if has is True]
            body_ids: list[int] = [eid for eid, has in cap_results if has is False]

            # Reserve 40% for face shots, 25% for body-only, 35% flexible
            face_reserve = max_imgs * 40 // 100
            body_reserve = max_imgs * 25 // 100

            selected: list[int] = face_ids[:face_reserve]
            selected.extend(body_ids[:body_reserve])
            flex_remaining = max_imgs - len(selected)
            if flex_remaining > 0:
                used = set(selected)
                leftover = [e for e in face_ids[face_reserve:] + body_ids[body_reserve:] if e not in used]
                selected.extend(leftover[:flex_remaining])
            all_ids = selected
            logger.info(f"Balanced: {min(len(face_ids), face_reserve)} face + {min(len(body_ids), body_reserve)} body reserved, {len(all_ids)} total")
        else:
            logger.info(f"Commit: capping {len(all_ids)} images to {max_imgs} (category={cat})")
            keep_confirmed = confirmed_ids[:max_imgs]
            remaining = max_imgs - len(keep_confirmed)
            keep_new = new_ids[:remaining] if remaining > 0 else []
            all_ids = list(set(keep_confirmed + keep_new))

    # First: unlink ALL events previously assigned to this object
    await session.execute(
        select(Event).where(Event.named_object_id == object_id)
    )
    old_events = (await session.execute(
        select(Event).where(Event.named_object_id == object_id)
    )).scalars().all()

    for ev in old_events:
        if ev.id not in all_ids:
            ev.named_object_id = None
            ev.event_type = EventType.object_detected
            session.add(ev)

    # Reset model completely
    obj.embedding = None
    obj.body_embedding = None
    obj.reference_image_count = 0

    # Rebuild from confirmed images (parallel + batch-accumulate)
    # Batch-load all events and camera names
    if all_ids:
        commit_ev_result = await session.execute(select(Event).where(Event.id.in_(all_ids)))
        commit_ev_map = {ev.id: ev for ev in commit_ev_result.scalars().all()}
    else:
        commit_ev_map = {}

    commit_cam_ids = list({ev.camera_id for ev in commit_ev_map.values()})
    if commit_cam_ids:
        commit_cam_rows = await session.execute(select(Camera.id, Camera.name).where(Camera.id.in_(commit_cam_ids)))
        commit_cam_map = {row.id: row.name for row in commit_cam_rows}
    else:
        commit_cam_map = {}

    sem_commit = asyncio.Semaphore(5)

    async def _commit_extract(eid):
        ev = commit_ev_map.get(eid)
        if not ev or not ev.thumbnail_path:
            return None
        async with sem_commit:
            # Use bbox crop to isolate the target object
            img_bytes = _load_event_crop(ev)
            if not img_bytes:
                return None
            crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                return None
            # Quality gate — skip blurry/tiny crops
            qok, qreason = _assess_crop_quality(crop, cat)
            if not qok:
                logger.info("Deep retrain: skipping event %d — %s", eid, qreason)
                return None
            face_emb = None
            body_emb = None
            has_face = False
            if is_person and face_service.is_available:
                faces, crop = await _detect_face_robust(crop, ev.snapshot_path, ev.bbox)
                if faces:
                    has_face = True
                    face_emb = _get_face_embedding(faces[0], crop)
                if recognition_service.reid_available:
                    body_emb = await asyncio.to_thread(recognition_service.compute_reid_embedding, crop)
            else:
                face_emb = await asyncio.to_thread(recognition_service._compute_embedding_best, crop)
            cam_name = commit_cam_map.get(ev.camera_id, "Unknown")
            return {"eid": eid, "face_emb": face_emb, "body_emb": body_emb, "has_face": has_face, "cam_name": cam_name}

    commit_results = await asyncio.gather(*[_commit_extract(eid) for eid in all_ids])

    # Batch-accumulate embeddings (equal weighting, single normalization at end)
    face_sum = None
    body_sum = None
    trained_face = 0
    trained_body = 0
    cov_detections = []

    all_ids_set = set(all_ids)
    for r in commit_results:
        if r is None:
            continue
        if r["face_emb"]:
            arr = np.array(r["face_emb"], dtype=np.float64)
            face_sum = arr if face_sum is None else face_sum + arr
            trained_face += 1
        if r["body_emb"]:
            arr = np.array(r["body_emb"], dtype=np.float64)
            body_sum = arr if body_sum is None else body_sum + arr
            trained_body += 1
        cov_detections.append({"camera_name": r["cam_name"], "has_face": r["has_face"]})

        ev = commit_ev_map.get(r["eid"])
        if ev:
            ev.named_object_id = obj.id
            ev.event_type = EventType.object_recognized
            session.add(ev)
            obj.reference_image_count += 1

    # Compute final averaged embeddings
    if face_sum is not None and trained_face > 0:
        avg = face_sum / trained_face
        norm = float(np.linalg.norm(avg))
        if norm > 0:
            avg = avg / norm
        obj.embedding = avg.tolist()

    if body_sum is not None and trained_body > 0:
        avg = body_sum / trained_body
        norm = float(np.linalg.norm(avg))
        if norm > 0:
            avg = avg / norm
        obj.body_embedding = avg.tolist()

    commit_coverage = _compute_coverage(cov_detections, cat)

    session.add(obj)
    await session.commit()

    return {
        "object_id": obj.id,
        "object_name": obj.name,
        "total_confirmed": len(all_ids),
        "trained_face": trained_face,
        "trained_body": trained_body,
        "reference_count": obj.reference_image_count,
        "unlinked": sum(1 for ev in old_events if ev.id not in all_ids),
        "max_images": max_imgs,
        "capped": len(data.get("confirmed_event_ids", []) + data.get("new_event_ids", [])) > max_imgs,
        "coverage": commit_coverage,
    }


async def _train_object_with_bytes(obj: NamedObject, img_bytes: bytes) -> dict:
    """Train a named object with image bytes.

    For persons: detect face → extract 512-dim ArcFace embedding + body ReID.
    For others: MobileNetV2 CNN embedding.

    Returns dict: {"ok": bool, "face_trained": bool, "body_only": bool, "skip_reason": str|None}
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if crop is None:
        return {"ok": False, "face_trained": False, "body_only": False, "skip_reason": "decode failed"}

    cat = obj.category.value if hasattr(obj.category, "value") else obj.category
    qok, qreason = _assess_crop_quality(crop, cat)
    if not qok:
        logger.info("Skipping low-quality training crop for '%s': %s", obj.name, qreason)
        return {"ok": False, "face_trained": False, "body_only": False, "skip_reason": qreason}

    if obj.category == ObjectCategory.person and face_service.is_available:
        faces = await face_service.detect_faces_async(crop)
        if not faces:
            # Upscale retry for small images
            h, w = crop.shape[:2]
            if h < 200 or w < 120:
                scale = min(2.5, max(200 / h, 120 / w))
                upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                faces = await face_service.detect_faces_async(upscaled)
                if faces:
                    crop = upscaled
        if not faces:
            # No face found — still compute body embedding if ReID available
            body_emb = await recognition_service.compute_reid_embedding_async(crop)
            if body_emb:
                obj.body_embedding = recognition_service.merge_reid_embedding(
                    obj.body_embedding, body_emb, obj.reference_image_count
                )
                logger.warning("No face detected for '%s' — trained body-only from this image", obj.name)
                return {"ok": True, "face_trained": False, "body_only": True, "skip_reason": None}
            logger.warning("No face in training image for person '%s' — skipping", obj.name)
            return {"ok": False, "face_trained": False, "body_only": False, "skip_reason": "no face detected"}
        emb = _get_face_embedding(faces[0], crop)
        if not emb:
            return {"ok": False, "face_trained": False, "body_only": False, "skip_reason": "face embedding extraction failed"}
        obj.embedding = face_service.merge_face_embeddings(
            obj.embedding, emb, obj.reference_image_count
        )
        # Also compute body ReID embedding from the full crop
        body_emb = await recognition_service.compute_reid_embedding_async(crop)
        if body_emb:
            obj.body_embedding = recognition_service.merge_reid_embedding(
                obj.body_embedding, body_emb, obj.reference_image_count
            )
        return {"ok": True, "face_trained": True, "body_only": False, "skip_reason": None}

    obj.embedding = await recognition_service.compute_and_merge_embedding_async(
        crop, obj.embedding, obj.reference_image_count
    )
    return {"ok": True, "face_trained": False, "body_only": False, "skip_reason": None}


# ──────────────── Auto-Clustering of Unnamed Detections ──────────────────

@router.get("/clusters")
async def get_unnamed_clusters(
    object_type: Optional[str] = Query(None),
    max_events: int = Query(500, ge=10, le=2000),
    similarity_threshold: float = Query(0.0, ge=0.0, le=1.0),
    session: AsyncSession = Depends(get_session),
):
    """Group recent unnamed detections into clusters by visual similarity.

    Returns a list of clusters, each with a representative thumbnail and
    a list of event IDs + thumbnails that belong to the cluster.
    """
    VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "bicycle", "boat", "train", "airplane"]
    PET_TYPES = ["cat", "dog"]

    filters = [
        Event.named_object_id.is_(None),
        Event.thumbnail_path.isnot(None),
        Event.event_type != EventType.motion,
        Event.metadata_extra.isnot(None),
    ]

    if object_type:
        if object_type == "pet":
            filters.append(Event.object_type.in_(PET_TYPES))
        elif object_type == "vehicle":
            filters.append(Event.object_type.in_(VEHICLE_TYPES))
        elif object_type == "other":
            filters.append(~Event.object_type.in_(["person"] + PET_TYPES + VEHICLE_TYPES))
        else:
            filters.append(Event.object_type == object_type)

    result = await session.execute(
        select(Event)
        .where(and_(*filters))
        .order_by(desc(Event.started_at))
        .limit(max_events)
    )
    events = result.scalars().all()

    # Embedding dimensions that are useful for identity clustering
    USABLE_DIMS = {512, 256, 1280}  # face=512, ReID=256, CNN=1280
    # 192 = colour histogram (clothing colour, not identity) — useless
    # 128 = stale SFace model — outdated

    # Build list of events with usable embeddings
    items: list[dict] = []
    skipped_bad_dim = 0
    camera_cache: dict[int, str] = {}
    for ev in events:
        emb = (ev.metadata_extra or {}).get("embedding")
        if not emb:
            continue
        dim = len(emb)
        if dim not in USABLE_DIMS:
            skipped_bad_dim += 1
            continue
        if ev.camera_id not in camera_cache:
            cam = await session.execute(select(Camera.name).where(Camera.id == ev.camera_id))
            camera_cache[ev.camera_id] = cam.scalar_one_or_none() or "Unknown"
        items.append({
            "event_id": ev.id,
            "camera_id": ev.camera_id,
            "camera_name": camera_cache[ev.camera_id],
            "object_type": ev.object_type,
            "confidence": ev.confidence,
            "thumbnail_url": f"/api/events/{ev.id}/crop",
            "snapshot_url": f"/api/events/{ev.id}/snapshot" if ev.snapshot_path else None,
            "timestamp": ev.started_at.isoformat(),
            "embedding": emb,
        })

    if not items:
        return {"clusters": [], "unclustered_count": 0, "skipped_bad_embeddings": skipped_bad_dim}

    # Auto-select threshold by embedding dimension if not explicitly set
    # These are per-dimension thresholds applied when the default threshold=0
    DIM_THRESHOLDS = {512: 0.40, 256: 0.72, 1280: 0.82}
    if similarity_threshold <= 0.01:
        sample_dim = len(items[0]["embedding"])
        similarity_threshold = DIM_THRESHOLDS.get(sample_dim, 0.78)

    # Centroid-based greedy clustering with drift protection
    clusters: list[dict] = []
    cluster_centroids: list[np.ndarray] = []
    cluster_reps: list[np.ndarray] = []  # original representative embedding
    assigned = set()

    # Pre-compute normalised embeddings
    emb_arrays = []
    for item in items:
        arr = np.array(item["embedding"], dtype=np.float32)
        norm = np.linalg.norm(arr)
        emb_arrays.append(arr / norm if norm > 0 else arr)

    def _matches_cluster(emb: np.ndarray, ci: int) -> float:
        """Return similarity if emb matches cluster ci (both centroid AND rep), else -1."""
        centroid = cluster_centroids[ci]
        if len(centroid) != len(emb):
            return -1.0
        sim_centroid = float(np.dot(emb, centroid))
        sim_rep = float(np.dot(emb, cluster_reps[ci]))
        # Must exceed threshold against both centroid and representative
        if sim_centroid < similarity_threshold or sim_rep < similarity_threshold:
            return -1.0
        return min(sim_centroid, sim_rep)

    for i in range(len(items)):
        if i in assigned:
            continue

        # Try to add to an existing cluster
        best_cluster = -1
        best_sim = similarity_threshold
        for ci in range(len(clusters)):
            if clusters[ci]["object_type"] != items[i]["object_type"]:
                continue
            sim = _matches_cluster(emb_arrays[i], ci)
            if sim > best_sim:
                best_sim = sim
                best_cluster = ci

        if best_cluster >= 0:
            # Add to existing cluster, update centroid
            clusters[best_cluster]["events"].append(items[i])
            clusters[best_cluster]["size"] += 1
            n = clusters[best_cluster]["size"]
            old_c = cluster_centroids[best_cluster]
            new_c = old_c + (emb_arrays[i] - old_c) / n
            norm = np.linalg.norm(new_c)
            cluster_centroids[best_cluster] = new_c / norm if norm > 0 else new_c
            assigned.add(i)
        else:
            # Start new cluster
            cluster_members = [items[i]]
            rep = emb_arrays[i].copy()
            centroid = emb_arrays[i].copy()
            assigned.add(i)

            # Sweep remaining to seed cluster
            for j in range(i + 1, len(items)):
                if j in assigned:
                    continue
                if items[j]["object_type"] != items[i]["object_type"]:
                    continue
                if len(emb_arrays[j]) != len(emb_arrays[i]):
                    continue
                sim_centroid = float(np.dot(emb_arrays[j], centroid))
                sim_rep = float(np.dot(emb_arrays[j], rep))
                if sim_centroid >= similarity_threshold and sim_rep >= similarity_threshold:
                    cluster_members.append(items[j])
                    assigned.add(j)
                    n = len(cluster_members)
                    centroid = centroid + (emb_arrays[j] - centroid) / n
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm

            clusters.append({
                "cluster_id": len(clusters),
                "object_type": items[i]["object_type"],
                "size": len(cluster_members),
                "representative": cluster_members[0],
                "events": cluster_members,
            })
            cluster_centroids.append(centroid)
            cluster_reps.append(rep)

    # Strip embeddings from response
    for cl in clusters:
        for m in cl["events"]:
            del m["embedding"]

    # Sort: biggest clusters first, then singletons
    clusters.sort(key=lambda c: c["size"], reverse=True)
    # Re-number after sort
    for i, cl in enumerate(clusters):
        cl["cluster_id"] = i

    unclustered = sum(1 for c in clusters if c["size"] == 1)

    return {
        "clusters": clusters,
        "total_clustered": len(items),
        "unclustered_count": unclustered,
        "skipped_bad_embeddings": skipped_bad_dim,
    }


@router.post("/clusters/backfill")
async def backfill_cluster_embeddings(
    object_type: str = Query("person"),
    batch_size: int = Query(100, ge=10, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Compute embeddings for orphaned events (e.g. after deleting named objects).

    These events had no embedding stored because they were originally recognised.
    This endpoint reads their thumbnails and computes embeddings so they can cluster.
    """
    PET_TYPES = ["cat", "dog"]
    VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "bicycle", "boat", "train", "airplane"]

    filters = [
        Event.named_object_id.is_(None),
        Event.thumbnail_path.isnot(None),
        Event.event_type != EventType.motion,
    ]

    if object_type == "pet":
        filters.append(Event.object_type.in_(PET_TYPES))
    elif object_type == "vehicle":
        filters.append(Event.object_type.in_(VEHICLE_TYPES))
    elif object_type == "other":
        filters.append(~Event.object_type.in_(["person"] + PET_TYPES + VEHICLE_TYPES))
    else:
        filters.append(Event.object_type == object_type)

    # Find events without embeddings
    result = await session.execute(
        select(Event)
        .where(
            and_(
                *filters,
                Event.metadata_extra.is_(None)
                | ~Event.metadata_extra.has_key("embedding"),
            )
        )
        .order_by(desc(Event.started_at))
        .limit(batch_size)
    )
    events = result.scalars().all()

    filled = 0
    failed = 0
    for ev in events:
        try:
            thumb_path = Path(ev.thumbnail_path)
            if not thumb_path.exists():
                failed += 1
                continue
            img = cv2.imread(str(thumb_path))
            if img is None:
                failed += 1
                continue

            if object_type == "person":
                # Try face embedding first (512-dim ArcFace — best for identity)
                faces = await face_service.detect_faces_async(img)
                if faces and faces[0].embedding:
                    emb = faces[0].embedding
                    meta = ev.metadata_extra or {}
                    meta["embedding"] = emb
                    ev.metadata_extra = meta
                    session.add(ev)
                    filled += 1
                    continue
                # Fallback: body ReID (256-dim — good for identity)
                reid_emb = await recognition_service.compute_reid_embedding_async(img)
                if reid_emb:
                    meta = ev.metadata_extra or {}
                    meta["embedding"] = reid_emb if isinstance(reid_emb, list) else reid_emb.tolist()
                    ev.metadata_extra = meta
                    session.add(ev)
                    filled += 1
                    continue
                # Skip — don't store useless histogram/CNN embeddings for persons
                failed += 1
                continue
            else:
                emb = await recognition_service.compute_embedding(img)

            meta = ev.metadata_extra or {}
            meta["embedding"] = emb
            ev.metadata_extra = meta
            session.add(ev)
            filled += 1
        except Exception as e:
            logger.debug("Backfill failed for event %d: %s", ev.id, e)
            failed += 1

    await session.commit()

    # Count remaining
    remaining_result = await session.execute(
        select(func.count(Event.id))
        .where(
            and_(
                *filters,
                Event.metadata_extra.is_(None)
                | ~Event.metadata_extra.has_key("embedding"),
            )
        )
    )
    remaining = remaining_result.scalar_one()

    return {
        "filled": filled,
        "failed": failed,
        "remaining": remaining,
    }


@router.post("/clusters/rebackfill")
async def rebackfill_bad_embeddings(
    object_type: str = Query("person"),
    batch_size: int = Query(100, ge=10, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Replace low-quality embeddings (192-dim histogram, 128-dim old SFace) with face/ReID.

    These embeddings cannot distinguish identity. This endpoint replaces them
    with 512-dim face embeddings or 256-dim body ReID embeddings.
    """
    BAD_DIMS = {192, 128}

    filters = [
        Event.named_object_id.is_(None),
        Event.thumbnail_path.isnot(None),
        Event.event_type != EventType.motion,
        Event.metadata_extra.isnot(None),
        Event.metadata_extra.has_key("embedding"),
    ]
    if object_type == "person":
        filters.append(Event.object_type == "person")
    else:
        return {"error": "re-backfill only supported for person type"}

    result = await session.execute(
        select(Event)
        .where(and_(*filters))
        .order_by(desc(Event.started_at))
        .limit(batch_size * 5)  # over-fetch since we filter by dim in Python
    )
    events = result.scalars().all()

    # Filter to only events with bad embedding dimensions
    bad_events = []
    for ev in events:
        emb = (ev.metadata_extra or {}).get("embedding")
        if emb and len(emb) in BAD_DIMS:
            bad_events.append(ev)
        if len(bad_events) >= batch_size:
            break

    replaced = 0
    failed = 0
    for ev in bad_events:
        try:
            thumb_path = Path(ev.thumbnail_path)
            if not thumb_path.exists():
                failed += 1
                continue
            img = cv2.imread(str(thumb_path))
            if img is None:
                failed += 1
                continue

            # Try face embedding first
            faces = await face_service.detect_faces_async(img)
            if faces and faces[0].embedding:
                meta = ev.metadata_extra or {}
                meta["embedding"] = faces[0].embedding
                ev.metadata_extra = meta
                session.add(ev)
                replaced += 1
                continue

            # Fallback: body ReID
            reid_emb = await recognition_service.compute_reid_embedding_async(img)
            if reid_emb:
                meta = ev.metadata_extra or {}
                meta["embedding"] = reid_emb if isinstance(reid_emb, list) else reid_emb.tolist()
                ev.metadata_extra = meta
                session.add(ev)
                replaced += 1
                continue

            failed += 1
        except Exception as e:
            logger.debug("Re-backfill failed for event %d: %s", ev.id, e)
            failed += 1

    await session.commit()

    # Count remaining bad embeddings
    total_bad = sum(
        1 for ev in events
        if (ev.metadata_extra or {}).get("embedding")
        and len((ev.metadata_extra or {}).get("embedding", [])) in BAD_DIMS
    )

    return {
        "replaced": replaced,
        "failed": failed,
        "remaining_bad": max(0, total_bad - replaced),
    }


# ──────────────── Re-extract from Recordings ──────────────────

# Background task state — survives screen lock / disconnect
_reextract_state: dict = {
    "running": False,
    "phase": "idle",           # idle | scanning | extracting | done | error
    "total": 0,
    "low_res": 0,
    "current": 0,
    "updated": 0,
    "faces_found": 0,
    "failed": 0,
    "skipped": 0,
    "error": None,
}


def _check_needs_reextract(ev, object_type: str) -> bool:
    """Return True if this event's thumbnail needs re-extraction."""
    max_width = 480 if object_type == "person" else 200
    threshold = max_width - 20  # allow small tolerance

    if not ev.thumbnail_path:
        return True
    tp = Path(ev.thumbnail_path)
    if not tp.exists():
        return True
    try:
        img = cv2.imdecode(np.frombuffer(tp.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return True
        return img.shape[1] < threshold
    except Exception:
        return True


async def _run_reextract(events: list, object_type: str):
    """Background coroutine that does the actual extraction work."""
    global _reextract_state
    try:
        _reextract_state.update(phase="scanning", current=0, updated=0,
                                faces_found=0, failed=0, skipped=0, error=None)

        # Filter to only events that need re-extraction
        needs_work = []
        for ev in events:
            if _check_needs_reextract(ev, object_type):
                needs_work.append(ev)

        _reextract_state.update(
            phase="extracting",
            total=len(needs_work),
            low_res=len(needs_work),
        )

        if not needs_work:
            _reextract_state.update(phase="done", total=0, low_res=0)
            return

        for i, ev in enumerate(needs_work):
            try:
                frame = await asyncio.to_thread(
                    _extract_frame_from_segments, ev.camera_id, ev.started_at
                )
                if frame is None:
                    _reextract_state["failed"] += 1
                    _reextract_state["current"] = i + 1
                    continue

                bbox = ev.bbox
                fh, fw = frame.shape[:2]
                x1 = max(0, int(bbox.get("x1", 0)))
                y1 = max(0, int(bbox.get("y1", 0)))
                x2 = min(fw, int(bbox.get("x2", fw)))
                y2 = min(fh, int(bbox.get("y2", fh)))

                if x2 <= x1 or y2 <= y1:
                    _reextract_state["failed"] += 1
                else:
                    # Centred crop with generous padding
                    box_w, box_h = x2 - x1, y2 - y1
                    if object_type == "person":
                        pad_top = int(box_h * 0.60)
                        pad_bottom = int(box_h * 0.30)
                        pad_left = int(box_w * 0.50)
                        pad_right = int(box_w * 0.50)
                    else:
                        pad = int(max(box_w, box_h) * 0.40)
                        pad_top = pad_bottom = pad_left = pad_right = pad
                    cx1, cy1 = x1 - pad_left, y1 - pad_top
                    cx2, cy2 = x2 + pad_right, y2 + pad_bottom
                    if cx1 < 0: cx2 = min(fw, cx2 - cx1); cx1 = 0
                    if cx2 > fw: cx1 = max(0, cx1 - (cx2 - fw)); cx2 = fw
                    if cy1 < 0: cy2 = min(fh, cy2 - cy1); cy1 = 0
                    if cy2 > fh: cy1 = max(0, cy1 - (cy2 - fh)); cy2 = fh
                    cx1, cy1 = max(0, cx1), max(0, cy1)
                    cx2, cy2 = min(fw, cx2), min(fh, cy2)
                    if cx2 <= cx1 or cy2 <= cy1:
                        crop = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)].copy()
                    else:
                        crop = frame[cy1:cy2, cx1:cx2].copy()
                    max_width = 480 if object_type == "person" else 200
                    h, w = crop.shape[:2]
                    save_crop = crop
                    if w > max_width:
                        scale = max_width / w
                        save_crop = cv2.resize(crop, (max_width, int(h * scale)))

                    if ev.thumbnail_path:
                        thumb_path = Path(ev.thumbnail_path)
                        thumb_path.parent.mkdir(parents=True, exist_ok=True)
                        await asyncio.to_thread(
                            cv2.imwrite, str(thumb_path), save_crop, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )

                    if ev.snapshot_path:
                        snap_path = Path(ev.snapshot_path)
                        clean_path = snap_path.parent / snap_path.name.replace(".jpg", "_clean.jpg")
                        if not clean_path.exists():
                            await asyncio.to_thread(
                                cv2.imwrite, str(clean_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                            )

                    # Rebuild consolidation GIF if this event has thumbnail history
                    meta = ev.metadata_extra or {}
                    thumb_history = meta.get("thumbnail_history")
                    if thumb_history and len(thumb_history) >= 2:
                        try:
                            from services.frigate_bridge import frigate_bridge
                            gif_path = await frigate_bridge._build_preview_gif(
                                ev.camera_id, thumb_history, ev.object_type or object_type,
                            )
                            if gif_path:
                                old_gif = meta.get("gif_path")
                                if old_gif and old_gif != gif_path:
                                    try:
                                        Path(old_gif).unlink(missing_ok=True)
                                    except Exception:
                                        pass
                                meta["gif_path"] = gif_path
                                ev.metadata_extra = meta
                                async with async_session() as sess:
                                    db_ev = await sess.get(Event, ev.id)
                                    if db_ev:
                                        db_ev.metadata_extra = meta
                                        await sess.commit()
                        except Exception as gif_err:
                            logger.debug("GIF rebuild failed for event %d: %s", ev.id, gif_err)

                    faces, _ = await _detect_face_robust(crop, ev.snapshot_path, bbox)
                    if faces:
                        _reextract_state["faces_found"] += 1

                    _reextract_state["updated"] += 1

            except Exception as e:
                logger.debug("Re-extract failed for event %d: %s", ev.id, e)
                _reextract_state["failed"] += 1

            _reextract_state["current"] = i + 1

        _reextract_state["phase"] = "done"

    except Exception as e:
        logger.error("Re-extract background task error: %s", e)
        _reextract_state.update(phase="error", error=str(e))
    finally:
        _reextract_state["running"] = False


@router.post("/re-extract-thumbnails")
async def re_extract_thumbnails(
    hours: int = Query(default=24, ge=1, le=168, description="Look back this many hours"),
    object_type: str = Query(default="person", description="Object type to re-extract"),
    session: AsyncSession = Depends(get_session),
):
    """Start background re-extraction of low-res thumbnails from HLS recordings.

    Only processes events whose thumbnails are below the target resolution.
    Runs as a background task — poll GET /re-extract-thumbnails/status for progress.
    Survives screen lock / browser disconnect.
    """
    global _reextract_state
    if _reextract_state["running"]:
        raise HTTPException(status_code=409, detail="Re-extraction already in progress")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    result = await session.execute(
        select(Event)
        .where(
            and_(
                Event.object_type == object_type,
                Event.started_at >= cutoff,
                Event.bbox.isnot(None),
            )
        )
        .order_by(desc(Event.started_at))
        .limit(500)
    )
    events = list(result.scalars().all())

    _reextract_state.update(
        running=True, phase="scanning", total=len(events), low_res=0,
        current=0, updated=0, faces_found=0, failed=0, skipped=0, error=None,
    )

    asyncio.create_task(_run_reextract(events, object_type))

    return {"status": "started", "total_events": len(events)}


@router.get("/re-extract-thumbnails/status")
async def re_extract_status():
    """Poll current re-extraction progress. Frontend calls this every ~2 seconds."""
    return dict(_reextract_state)


def _extract_frame_from_segments(camera_id: int, target_dt: datetime) -> Optional[np.ndarray]:
    """Extract a full-res frame from HLS recordings closest to target_dt.

    Finds the hour directory, concatenates nearby segments into a short clip,
    and seeks to the target time offset to extract a single frame.
    """
    hour_dir = (
        Path(settings.recordings_path)
        / str(camera_id)
        / target_dt.strftime("%Y-%m-%d")
        / target_dt.strftime("%H")
    )
    if not hour_dir.exists():
        return None

    segments = sorted(hour_dir.glob("segment-*.ts"))
    if not segments:
        return None

    # Each segment is ~4 seconds. Target offset within the hour.
    seconds_into_hour = target_dt.minute * 60 + target_dt.second
    approx_seg_idx = seconds_into_hour // 4

    # Take a few segments around the target for better seek accuracy
    start_idx = max(0, approx_seg_idx - 2)
    end_idx = min(len(segments), approx_seg_idx + 3)
    nearby = segments[start_idx:end_idx]
    if not nearby:
        nearby = segments[:5]

    # Build concat list
    concat_content = "\n".join(f"file '{seg}'" for seg in nearby)
    concat_file = hour_dir / "_tmp_reextract.txt"
    try:
        concat_file.write_text(concat_content)

        # Offset within the concat clip
        offset_in_clip = max(0, (approx_seg_idx - start_idx) * 4)

        # Use FFmpeg to extract a single frame as raw image bytes (BMP for lossless)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-ss", str(offset_in_clip),
            "-frames:v", "1",
            "-f", "image2pipe",
            "-vcodec", "bmp",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=30)
        if proc.returncode != 0:
            logger.warning("FFmpeg frame extract failed: %s", proc.stderr.decode(errors="replace")[-300:])
            return None

        frame_data = proc.stdout
        if not frame_data:
            return None

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error("Frame extraction error: %s", e)
        return None
    finally:
        concat_file.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Profile Integrity Analysis
# ═══════════════════════════════════════════════════════════════════════

async def _load_profile_thumbnails(obj_id: int, limit: int = 12) -> list[np.ndarray]:
    """Load recent high-confidence event thumbnails for a named object."""
    async with async_session() as session:
        result = await session.execute(
            select(Event.thumbnail_path)
            .where(
                Event.named_object_id == obj_id,
                Event.thumbnail_path.isnot(None),
                Event.confidence >= 0.50,
            )
            .order_by(Event.started_at.desc())
            .limit(limit * 2)
        )
        paths = result.scalars().all()

    crops = []
    for p in paths:
        if len(crops) >= limit:
            break
        try:
            resolved = _resolve_path(p)
            if resolved and resolved.exists():
                img = cv2.imread(str(resolved))
                if img is not None and img.size > 0:
                    crops.append(img)
        except Exception:
            pass
    return crops


async def check_profile_integrity(obj: NamedObject) -> Optional[dict]:
    """Run vision LLM analysis on a profile's event thumbnails to detect mismatches.

    Returns dict: {consistent, outlier_indices, confidence, reasoning, checked_images}
    or None if the check couldn't run (ML unavailable, not enough images).
    """
    from services.ml_client import remote_profile_analyze

    thumbnails = await _load_profile_thumbnails(obj.id, limit=12)
    if len(thumbnails) < 3:
        return None

    cat = obj.category.value if hasattr(obj.category, "value") else obj.category
    result = await remote_profile_analyze(
        thumbnails,
        profile_name=obj.name,
        object_type=cat,
    )
    if result is None:
        return None

    result["checked_images"] = len(thumbnails)
    return result


async def run_all_profile_checks() -> list[dict]:
    """Run integrity checks on all profiles with enough event history.

    Returns list of {id, name, category, consistent, outlier_indices, confidence, reasoning}.
    """
    results = []
    async with async_session() as session:
        all_objs = (await session.execute(
            select(NamedObject).order_by(NamedObject.name)
        )).scalars().all()

    for obj in all_objs:
        try:
            check = await check_profile_integrity(obj)
            if check is None:
                continue
            entry = {
                "id": obj.id,
                "name": obj.name,
                "category": obj.category.value if hasattr(obj.category, "value") else obj.category,
                **check,
            }
            results.append(entry)

            # Persist result in attributes
            async with async_session() as session:
                db_obj = (await session.execute(
                    select(NamedObject).where(NamedObject.id == obj.id)
                )).scalar_one_or_none()
                if db_obj:
                    attrs = dict(db_obj.attributes or {})
                    attrs["integrity"] = {
                        "consistent": check["consistent"],
                        "outlier_indices": check.get("outlier_indices", []),
                        "confidence": check.get("confidence", 0),
                        "reasoning": check.get("reasoning", "")[:200],
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                        "checked_images": check.get("checked_images", 0),
                    }
                    db_obj.attributes = attrs
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(db_obj, "attributes")
                    session.add(db_obj)
                    await session.commit()

            logger.info(
                "Profile integrity [%s]: consistent=%s outliers=%s conf=%d",
                obj.name, check["consistent"], check.get("outlier_indices", []),
                check.get("confidence", 0),
            )
        except Exception:
            logger.warning("Profile check failed for %s", obj.name, exc_info=True)

    return results


@router.post("/objects/{object_id}/integrity-check")
async def check_object_integrity(object_id: int, session: AsyncSession = Depends(get_session)):
    """Manually trigger a profile integrity check using vision AI."""
    result = await session.execute(select(NamedObject).where(NamedObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Object not found")

    check = await check_profile_integrity(obj)
    if check is None:
        raise HTTPException(
            status_code=503,
            detail="Could not run integrity check — ML server unavailable or not enough images (need 3+)"
        )

    # Persist result
    attrs = dict(obj.attributes or {})
    attrs["integrity"] = {
        "consistent": check["consistent"],
        "outlier_indices": check.get("outlier_indices", []),
        "confidence": check.get("confidence", 0),
        "reasoning": check.get("reasoning", "")[:200],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checked_images": check.get("checked_images", 0),
    }
    obj.attributes = attrs
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(obj, "attributes")
    session.add(obj)
    await session.commit()

    return check


@router.post("/integrity-check-all")
async def check_all_integrity():
    """Run profile integrity checks on all named objects. Returns results for profiles with issues."""
    results = await run_all_profile_checks()
    flagged = [r for r in results if not r.get("consistent", True)]
    return {
        "total_checked": len(results),
        "flagged": len(flagged),
        "results": results,
    }


# ──────────────── Cross-Profile Mismatch Review ──────────────────


@router.post("/cross-audit")
async def cross_audit():
    """Audit ALL profiles at once — find detections that may be assigned to the wrong profile.

    Returns profile list (with thumbnail URLs) and flagged detections across all profiles,
    sorted by similarity (worst first) so a human can review and reassign.
    """
    async with async_session() as session:
        all_objs = (await session.execute(
            select(NamedObject).where(NamedObject.embedding.isnot(None)).order_by(NamedObject.name)
        )).scalars().all()

    profiles = []
    all_flagged = []

    for obj in all_objs:
        cat = obj.category.value if hasattr(obj.category, "value") else obj.category

        # Get best profile thumbnail from most recent event
        async with async_session() as session:
            last_ev = (await session.execute(
                select(Event.id)
                .where(and_(Event.named_object_id == obj.id, Event.thumbnail_path.isnot(None)))
                .order_by(desc(Event.started_at))
                .limit(1)
            )).scalar_one_or_none()

        thumb_url = f"/api/events/{last_ev}/crop" if last_ev else None

        profiles.append({
            "id": obj.id,
            "name": obj.name,
            "category": cat,
            "thumbnail_url": thumb_url,
            "attributes": {
                k: v for k, v in (obj.attributes or {}).items()
                if not k.startswith("_") and not k.endswith("_confidence")
                and not k.endswith("_samples") and k != "integrity"
            },
        })

        stored_array = np.array(obj.embedding, dtype=np.float32)
        stored_norm = float(np.linalg.norm(stored_array))
        is_face = obj.category == ObjectCategory.person and len(obj.embedding) == FACE_EMBED_DIM

        async with async_session() as session:
            ev_result = await session.execute(
                select(Event)
                .where(and_(Event.named_object_id == obj.id, Event.thumbnail_path.isnot(None)))
                .order_by(desc(Event.started_at))
                .limit(100)
            )
            events = ev_result.scalars().all()

        sims = []
        det_list = []
        for ev in events:
            img_bytes = _load_event_crop(ev)
            if not img_bytes:
                continue
            crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                continue

            if is_face and face_service.is_available:
                faces = await face_service.detect_faces_async(crop)
                if faces:
                    emb = _get_face_embedding(faces[0], crop)
                    sim = face_service.cosine_similarity(emb, obj.embedding) if emb else 0.0
                else:
                    sim = 0.0
            else:
                emb = recognition_service._compute_embedding_best(crop)
                emb_array = np.array(emb, dtype=np.float32)
                if len(emb_array) != len(stored_array):
                    continue
                sim = float(np.dot(emb_array, stored_array)) / max(stored_norm * float(np.linalg.norm(emb_array)), 1e-8)

            sims.append(sim)
            async with async_session() as session:
                cam_name = (await session.scalar(select(Camera.name).where(Camera.id == ev.camera_id))) or "Unknown"
            det_list.append({
                "event_id": ev.id,
                "thumbnail_url": f"/api/events/{ev.id}/crop",
                "assigned_to_id": obj.id,
                "assigned_to_name": obj.name,
                "similarity": round(sim, 4),
                "camera_name": cam_name,
                "timestamp": ev.started_at.isoformat(),
            })

        if len(sims) < 3:
            continue

        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims)) if len(sims) > 1 else 0.0
        threshold = max(mean_sim - 2 * std_sim, 0.55)

        for d in det_list:
            if d["similarity"] < threshold:
                d["threshold"] = round(threshold, 4)
                all_flagged.append(d)

    all_flagged.sort(key=lambda d: d["similarity"])

    return {
        "profiles": profiles,
        "flagged": all_flagged,
        "total_audited": sum(1 for _ in all_objs),
    }


@router.post("/reassign-detections")
async def reassign_detections(
    data: dict,
    session: AsyncSession = Depends(get_session),
):
    """Bulk reassign or remove detections based on mismatch review.

    Body: { "actions": [ {"event_id": 123, "new_object_id": 5}, {"event_id": 456, "new_object_id": null} ] }
    new_object_id=null means remove from current profile (mark as unrecognized).
    """
    actions = data.get("actions", [])
    if not actions:
        raise HTTPException(status_code=400, detail="No actions provided")

    reassigned = 0
    removed = 0
    affected_object_ids: set[int] = set()

    for action in actions:
        event_id = action.get("event_id")
        new_object_id = action.get("new_object_id")

        ev = (await session.execute(select(Event).where(Event.id == event_id))).scalar_one_or_none()
        if not ev:
            continue

        old_obj_id = ev.named_object_id
        if old_obj_id:
            affected_object_ids.add(old_obj_id)

        if new_object_id is None:
            # Remove from profile (skip if pinned)
            if _is_pinned(ev):
                continue
            ev.named_object_id = None
            ev.event_type = EventType.object_detected
            session.add(ev)
            removed += 1
        else:
            # Reassign to different profile
            new_obj = (await session.execute(
                select(NamedObject).where(NamedObject.id == new_object_id)
            )).scalar_one_or_none()
            if not new_obj:
                continue
            ev.named_object_id = new_object_id
            ev.event_type = EventType.object_recognized
            await _pin_event(ev, reason="manual")
            session.add(ev)
            affected_object_ids.add(new_object_id)
            reassigned += 1

    await session.commit()

    # Recompute embeddings for affected profiles
    recomputed = []
    for obj_id in affected_object_ids:
        obj = (await session.execute(
            select(NamedObject).where(NamedObject.id == obj_id)
        )).scalar_one_or_none()
        if not obj:
            continue

        remaining_result = await session.execute(
            select(Event)
            .where(and_(Event.named_object_id == obj_id, Event.thumbnail_path.isnot(None)))
            .order_by(Event.started_at)
            .limit(200)
        )
        remaining = remaining_result.scalars().all()

        new_embedding = None
        count = 0
        for ev in remaining:
            img_bytes = _load_event_crop(ev)
            if not img_bytes:
                continue
            crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if crop is None:
                continue
            if obj.category == ObjectCategory.person and face_service.is_available:
                faces = await face_service.detect_faces_async(crop)
                if faces:
                    emb = _get_face_embedding(faces[0], crop)
                    if emb:
                        new_embedding = face_service.merge_face_embeddings(new_embedding, emb, count)
                        count += 1
            else:
                new_embedding = await recognition_service.compute_and_merge_embedding_async(crop, new_embedding, count)
                count += 1

        obj.embedding = new_embedding
        obj.reference_image_count = count
        session.add(obj)
        recomputed.append(obj.name)

    await session.commit()
    frigate_bridge.invalidate_embedding_cache()

    return {
        "reassigned": reassigned,
        "removed": removed,
        "recomputed_profiles": recomputed,
    }

