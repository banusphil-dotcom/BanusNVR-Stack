"""BanusNas — Smart Search API: last seen, timeline, activity queries, deep hunt."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user, get_user_from_token_param
from models.database import get_session, async_session
from models.schemas import Camera, Event, NamedObject, ObjectCategory
from schemas.api_schemas import LastSeenResponse, NamedObjectResponse, TimelineEntry
from services.frigate_bridge import frigate_bridge

logger = logging.getLogger(__name__)

# v14 pipeline overhaul date — profiles trained before this had a broken pre-filter
_V14_CUTOFF = datetime(2026, 4, 8, tzinfo=timezone.utc)


def _compute_retrain_status(obj: NamedObject, recent_event_count: int = 0) -> tuple[bool, list[str]]:
    """Determine whether a profile needs retraining after the v14 pipeline overhaul.

    Returns (needs_retrain, reasons).
    """
    reasons: list[str] = []
    category = obj.category.value if hasattr(obj.category, "value") else obj.category
    has_embedding = obj.embedding is not None
    created_pre_v14 = obj.created_at.replace(tzinfo=timezone.utc) < _V14_CUTOFF if obj.created_at else False

    # Any profile with embeddings trained before v14 needs retraining
    if created_pre_v14 and has_embedding and obj.reference_image_count > 0:
        reasons.append("Trained under old pipeline — embeddings may be inaccurate")

    if category == "person":
        # Person missing body model
        if not obj.body_embedding:
            reasons.append("No body model — cannot match when face is not visible")
        # Very low training data
        if has_embedding and obj.reference_image_count < 5:
            reasons.append("Insufficient training images (< 5)")
    else:
        if has_embedding and obj.reference_image_count < 3:
            reasons.append("Very few training images")

    # No embedding at all
    if not has_embedding:
        reasons.append("No model trained yet")

    return (len(reasons) > 0, reasons)

router = APIRouter(prefix="/api/search", tags=["search"], dependencies=[Depends(get_current_user)])
# Separate router for endpoints that use query-param auth (e.g. <img src>)
public_router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("/named-objects-status")
async def named_objects_status(session: AsyncSession = Depends(get_session)):
    """Get all named objects with their current/last location and live status."""
    # Get all named objects
    result = await session.execute(
        select(NamedObject).order_by(NamedObject.name)
    )
    named_objects = result.scalars().all()

    # Build a map of currently present named objects from Frigate bridge
    currently_seen: dict[str, tuple[int, str]] = {}  # name → (camera_id, camera_name)
    cameras_result = await session.execute(select(Camera).where(Camera.enabled == True))
    cameras = {c.id: c for c in cameras_result.scalars().all()}
    frigate_to_cam = {f"camera_{c.id}": (c.id, c) for c in cameras.values()}

    # 1. Frigate bridge in-memory presence (60s TTL)
    presence = frigate_bridge.get_current_presence()
    for name, cam_id in presence.items():
        if cam_id in cameras:
            currently_seen[name] = (cam_id, cameras[cam_id].name)

    # 2. Frigate active events with sub_labels
    frigate_events = []
    try:
        frigate_events = await frigate_bridge.get_active_frigate_events()
        for fev in frigate_events:
            sub = fev.get("sub_label")
            if not sub:
                continue
            sub = sub[0] if isinstance(sub, list) else sub
            if sub:
                cam_key = fev.get("frigate_camera", "")
                if cam_key in frigate_to_cam:
                    cid, cobj = frigate_to_cam[cam_key]
                    # Always overwrite — newer Frigate event reflects the most
                    # recent camera the named object was seen on.
                    currently_seen[sub] = (cid, cobj.name)
    except Exception:
        pass

    # NOTE: Live status reflects ONLY actively-tracked presence (Frigate
    # bridge in-memory presence with 60s TTL + currently-active Frigate
    # events). DB-based fallbacks intentionally removed so the indicator
    # is accurate ("currently in the room"), not historical.

    items = []
    for obj in named_objects:
        is_live = obj.name in currently_seen
        live_camera_id = currently_seen[obj.name][0] if is_live else None
        live_camera_name = currently_seen[obj.name][1] if is_live else None

        # Get last event for this object
        last_camera_name = None
        last_camera_id = None
        last_seen_at = None
        snapshot_url = None
        thumbnail_url = None

        ev_result = await session.execute(
            select(Event)
            .where(Event.named_object_id == obj.id)
            .order_by(desc(Event.started_at))
            .limit(1)
        )
        last_event = ev_result.scalar_one_or_none()
        if last_event:
            last_seen_at = last_event.started_at.isoformat()
            last_camera_id = last_event.camera_id
            cam = cameras.get(last_event.camera_id)
            last_camera_name = cam.name if cam else "Unknown"
            if last_event.thumbnail_path:
                thumbnail_url = f"/api/events/{last_event.id}/thumbnail"
            elif last_event.snapshot_path:
                snapshot_url = f"/api/events/{last_event.id}/snapshot"

        needs_retrain, retrain_reasons = _compute_retrain_status(obj)

        items.append({
            "id": obj.id,
            "name": obj.name,
            "category": obj.category.value if hasattr(obj.category, "value") else obj.category,
            "reference_image_count": obj.reference_image_count,
            "attributes": obj.attributes,
            "is_live": is_live,
            "live_camera_id": live_camera_id,
            "live_camera_name": live_camera_name,
            "last_camera_id": last_camera_id,
            "last_camera_name": last_camera_name,
            "last_seen_at": last_seen_at,
            "snapshot_url": snapshot_url,
            "thumbnail_url": thumbnail_url,
            "needs_retrain": needs_retrain,
            "retrain_reasons": retrain_reasons,
        })

    return items


@router.get("/last-seen/{name}", response_model=LastSeenResponse)
async def last_seen(name: str, session: AsyncSession = Depends(get_session)):
    """Find where a named object was last seen."""
    # Look up named object
    result = await session.execute(
        select(NamedObject).where(NamedObject.name.ilike(f"%{name}%"))
    )
    named_obj = result.scalar_one_or_none()
    if not named_obj:
        raise HTTPException(status_code=404, detail=f"No known object matching '{name}'")

    # Find latest event
    event_result = await session.execute(
        select(Event)
        .where(Event.named_object_id == named_obj.id)
        .order_by(desc(Event.started_at))
        .limit(1)
    )
    event = event_result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail=f"'{name}' has not been seen yet")

    # Get camera info
    cam_result = await session.execute(select(Camera).where(Camera.id == event.camera_id))
    camera = cam_result.scalar_one_or_none()

    return LastSeenResponse(
        named_object=NamedObjectResponse.model_validate(named_obj),
        camera_name=camera.name if camera else "Unknown",
        camera_id=event.camera_id,
        timestamp=event.started_at,
        snapshot_url=f"/api/events/{event.id}/snapshot" if event.snapshot_path else None,
    )


@router.get("/timeline", response_model=list[TimelineEntry])
async def search_timeline(
    object_name: Optional[str] = None,
    object_type: Optional[str] = None,
    camera_id: Optional[int] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
):
    """Get chronological appearances of an object or type."""
    query = select(Event).order_by(desc(Event.started_at)).limit(limit)

    filters = []

    if object_name:
        named_result = await session.execute(
            select(NamedObject).where(NamedObject.name.ilike(f"%{object_name}%"))
        )
        named_obj = named_result.scalar_one_or_none()
        if named_obj:
            filters.append(Event.named_object_id == named_obj.id)
        else:
            return []

    if object_type:
        filters.append(Event.object_type == object_type)
    if camera_id:
        filters.append(Event.camera_id == camera_id)
    if from_date:
        filters.append(Event.started_at >= from_date)
    if to_date:
        filters.append(Event.started_at <= to_date)

    if filters:
        query = query.where(and_(*filters))

    result = await session.execute(query)
    events = result.scalars().all()

    entries = []
    for ev in events:
        cam = await session.execute(select(Camera.name).where(Camera.id == ev.camera_id))
        cam_name = cam.scalar_one_or_none() or "Unknown"
        meta = ev.metadata_extra or {}
        entries.append(TimelineEntry(
            event_id=ev.id,
            camera_id=ev.camera_id,
            camera_name=cam_name,
            timestamp=ev.started_at,
            confidence=ev.confidence,
            snapshot_url=f"/api/events/{ev.id}/snapshot" if ev.snapshot_path else None,
            narrative=meta.get("narrative"),
        ))

    return entries


@router.get("/where")
async def search_where(
    object_type: Optional[str] = None,
    camera_id: Optional[int] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """Search for detections by type, camera, and time range."""
    query = select(Event).order_by(desc(Event.started_at)).limit(limit)

    filters = []
    if object_type:
        filters.append(Event.object_type == object_type)
    if camera_id:
        filters.append(Event.camera_id == camera_id)
    if from_date:
        filters.append(Event.started_at >= from_date)
    if to_date:
        filters.append(Event.started_at <= to_date)

    # Exclude pure motion events
    filters.append(Event.object_type.isnot(None))

    if filters:
        query = query.where(and_(*filters))

    result = await session.execute(query)
    events = result.scalars().all()

    return [
        {
            "event_id": ev.id,
            "camera_id": ev.camera_id,
            "object_type": ev.object_type,
            "named_object_id": ev.named_object_id,
            "confidence": ev.confidence,
            "timestamp": ev.started_at.isoformat(),
            "snapshot_url": f"/api/events/{ev.id}/snapshot" if ev.snapshot_path else None,
        }
        for ev in events
    ]


@router.get("/frequent")
async def frequent_objects(
    camera_id: Optional[int] = None,
    days: int = Query(7, ge=1, le=90),
    session: AsyncSession = Depends(get_session),
):
    """Get most frequently seen objects."""
    from datetime import timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    query = (
        select(Event.object_type, func.count().label("count"))
        .where(and_(Event.started_at >= cutoff, Event.object_type.isnot(None)))
        .group_by(Event.object_type)
        .order_by(desc("count"))
        .limit(20)
    )

    if camera_id:
        query = query.where(Event.camera_id == camera_id)

    result = await session.execute(query)
    rows = result.all()

    return [{"object_type": row[0], "count": row[1]} for row in rows]


@router.get("/frequent-named")
async def frequent_named_objects(
    days: int = Query(30, ge=1, le=365),
    session: AsyncSession = Depends(get_session),
):
    """Get named objects ranked by detection count."""
    from datetime import timedelta, timezone

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    result = await session.execute(
        select(
            Event.named_object_id,
            func.count().label("count"),
            func.max(Event.started_at).label("last_seen"),
        )
        .where(and_(Event.named_object_id.isnot(None), Event.started_at >= cutoff))
        .group_by(Event.named_object_id)
        .order_by(desc("count"))
        .limit(50)
    )
    rows = result.all()

    items = []
    for row in rows:
        obj_result = await session.execute(
            select(NamedObject).where(NamedObject.id == row[0])
        )
        obj = obj_result.scalar_one_or_none()
        if obj:
            items.append({
                "object_id": obj.id,
                "name": obj.name,
                "category": obj.category.value if hasattr(obj.category, "value") else obj.category,
                "count": row[1],
                "last_seen": row[2].isoformat() if row[2] else None,
            })

    return items


@router.get("/activity")
async def activity_heatmap(
    camera_id: int,
    date: str = Query(..., description="Date (YYYY-MM-DD)"),
    session: AsyncSession = Depends(get_session),
):
    """Get hourly activity counts for a camera."""
    from datetime import timedelta

    start = datetime.fromisoformat(f"{date}T00:00:00+00:00")
    end = start + timedelta(days=1)

    result = await session.execute(
        select(
            func.extract("hour", Event.started_at).label("hour"),
            func.count().label("count"),
        )
        .where(and_(Event.camera_id == camera_id, Event.started_at >= start, Event.started_at < end))
        .group_by("hour")
        .order_by("hour")
    )
    rows = result.all()

    # Fill in missing hours
    activity = {int(row[0]): row[1] for row in rows}
    return [{"hour": h, "count": activity.get(h, 0)} for h in range(24)]


@router.get("/activity-summary")
async def activity_summary(
    session: AsyncSession = Depends(get_session),
):
    """Get hourly activity counts across all cameras for the last 24 hours."""
    from datetime import timedelta, timezone

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=24)

    result = await session.execute(
        select(
            func.extract("hour", Event.started_at).label("hour"),
            func.count().label("count"),
        )
        .where(Event.started_at >= start)
        .group_by("hour")
        .order_by("hour")
    )
    rows = result.all()

    activity = {int(row[0]): row[1] for row in rows}
    return [{"hour": h, "count": activity.get(h, 0)} for h in range(24)]


# ────────── Deep Hunt ──────────


@router.post("/deep-hunt")
async def start_deep_hunt(
    named_object_id: int = Query(..., description="ID of the named object to hunt for"),
    camera_ids: Optional[str] = Query(None, description="Comma-separated camera IDs (empty = all)"),
    hours: int = Query(24, ge=1, le=72, description="How many hours back to scan"),
    frame_interval: float = Query(2.0, ge=0.5, le=10.0, description="Seconds between sampled frames"),
    session: AsyncSession = Depends(get_session),
):
    """Start a deep hunt job that scans recordings for a specific object."""
    import uuid
    from services.deep_hunt import HuntJob, _jobs, run_hunt

    # Load target object
    result = await session.execute(
        select(NamedObject).where(NamedObject.id == named_object_id)
    )
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="Named object not found")
    if not target.embedding:
        raise HTTPException(status_code=400, detail=f"'{target.name}' has no trained embedding — train first")

    # Parse camera IDs
    if camera_ids:
        cam_ids = [int(c.strip()) for c in camera_ids.split(",") if c.strip().isdigit()]
    else:
        # All enabled cameras
        cam_result = await session.execute(select(Camera.id).where(Camera.enabled == True))
        cam_ids = [row[0] for row in cam_result.all()]

    if not cam_ids:
        raise HTTPException(status_code=400, detail="No cameras selected")

    # Determine target YOLO classes based on category
    category = target.category.value if hasattr(target.category, "value") else target.category
    class_map = {
        "pet": ["cat", "dog"],
        "person": ["person"],
        "vehicle": ["car", "motorcycle", "bus", "truck"],
    }
    target_classes = class_map.get(category, ["person", "cat", "dog", "car"])

    now = datetime.now(timezone.utc)
    job = HuntJob(
        job_id=str(uuid.uuid4()),
        target_name=target.name,
        target_id=target.id,
        category=category,
        camera_ids=cam_ids,
        start_time=now - timedelta(hours=hours),
        end_time=now,
        frame_interval=frame_interval,
    )
    _jobs[job.job_id] = job

    # Launch as background task
    asyncio.create_task(run_hunt(job, target.embedding, target_classes))

    return {
        "job_id": job.job_id,
        "target": target.name,
        "cameras": cam_ids,
        "hours": hours,
        "frame_interval": frame_interval,
        "status": "started",
    }


@router.get("/deep-hunt/{job_id}")
async def get_hunt_status(job_id: str):
    """Get the current status and results of a deep hunt job."""
    from services.deep_hunt import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Hunt job not found")

    return {
        "job_id": job.job_id,
        "target": job.target_name,
        "status": job.status,
        "progress": round(job.progress, 3),
        "segments_total": job.segments_total,
        "segments_done": job.segments_done,
        "frames_scanned": job.frames_scanned,
        "detections_total": job.detections_total,
        "detections_relevant": job.detections_relevant,
        "sightings_count": len(job.sightings),
        "error": job.error or None,
        "sightings": [
            {
                "timestamp": s.timestamp,
                "camera": s.camera_name,
                "confidence": round(s.confidence, 3),
                "det_confidence": round(s.det_confidence, 3),
                "class_name": s.class_name,
                "bbox": {"x1": s.bbox[0], "y1": s.bbox[1], "x2": s.bbox[2], "y2": s.bbox[3]},
                "thumbnail_url": f"/api/search/deep-hunt/{job.job_id}/thumb/{i}",
            }
            for i, s in enumerate(job.sightings)
        ],
    }


@router.get("/deep-hunt/{job_id}/stream")
async def stream_hunt_progress(job_id: str):
    """SSE stream of hunt progress and new sightings."""
    import json
    from services.deep_hunt import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Hunt job not found")

    async def event_generator():
        last_sighting_count = 0
        while True:
            j = get_job(job_id)
            if not j:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break

            # Send progress update
            yield f"data: {json.dumps({'type': 'progress', 'progress': round(j.progress, 3), 'segments_done': j.segments_done, 'segments_total': j.segments_total, 'sightings_count': len(j.sightings), 'status': j.status})}\n\n"

            # Send new sightings
            if len(j.sightings) > last_sighting_count:
                for i in range(last_sighting_count, len(j.sightings)):
                    s = j.sightings[i]
                    yield f"data: {json.dumps({'type': 'sighting', 'index': i, 'timestamp': s.timestamp, 'camera': s.camera_name, 'confidence': round(s.confidence, 3), 'class_name': s.class_name, 'thumbnail_url': f'/api/search/deep-hunt/{job_id}/thumb/{i}'})}\n\n"
                last_sighting_count = len(j.sightings)

            if j.status in ("completed", "cancelled", "error"):
                yield f"data: {json.dumps({'type': 'done', 'status': j.status, 'total_sightings': len(j.sightings), 'error': j.error or None})}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@public_router.get("/deep-hunt/{job_id}/thumb/{index}")
async def get_hunt_thumbnail(
    job_id: str,
    index: int,
    _user=Depends(get_user_from_token_param),
):
    """Serve a thumbnail image from a deep hunt sighting."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    from services.deep_hunt import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Hunt job not found")
    if index < 0 or index >= len(job.sightings):
        raise HTTPException(status_code=404, detail="Sighting not found")

    thumb_path = Path(job.sightings[index].frame_path)
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(str(thumb_path), media_type="image/jpeg")


@router.post("/deep-hunt/{job_id}/cancel")
async def cancel_hunt(job_id: str):
    """Cancel a running deep hunt job."""
    from services.deep_hunt import cancel_job

    if cancel_job(job_id):
        return {"status": "cancelling"}
    raise HTTPException(status_code=400, detail="Job not found or not running")


@router.get("/deep-hunt-jobs")
async def list_hunt_jobs(target_id: Optional[int] = None):
    """List recent deep hunt jobs, optionally filtered to a target object."""
    from services.deep_hunt import list_jobs

    jobs = list_jobs(target_id=target_id)
    return [
        {
            "job_id": j.job_id,
            "target": j.target_name,
            "target_id": j.target_id,
            "status": j.status,
            "progress": round(j.progress, 3),
            "segments_total": j.segments_total,
            "segments_done": j.segments_done,
            "frames_scanned": j.frames_scanned,
            "sightings_count": len(j.sightings),
            "created_at": j.created_at,
        }
        for j in jobs
    ]


@router.post("/deep-hunt/{job_id}/add-to-training")
async def add_hunt_sightings_to_training(
    job_id: str,
    sighting_indices: list[int] = Query(..., description="Indices of sightings to add"),
    session: AsyncSession = Depends(get_session),
):
    """Add selected hunt sightings to the target object's training model."""
    import cv2
    import numpy as np
    from services.deep_hunt import get_job
    from services.recognition_service import recognition_service
    from services.face_service import face_service

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Hunt job not found")

    # Load the target object
    result = await session.execute(
        select(NamedObject).where(NamedObject.id == job.target_id)
    )
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status_code=404, detail="Target object not found")

    added = 0
    for idx in sighting_indices:
        if idx < 0 or idx >= len(job.sightings):
            continue
        sighting = job.sightings[idx]
        thumb_path = Path(sighting.frame_path)
        if not thumb_path.exists():
            continue

        img_bytes = thumb_path.read_bytes()
        nparr = np.frombuffer(img_bytes, np.uint8)
        crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if crop is None:
            continue

        category = obj.category.value if hasattr(obj.category, "value") else obj.category
        if category == "person" and face_service.is_available:
            faces = await face_service.detect_faces_async(crop)
            if faces:
                from routers.training import _get_face_embedding
                emb = _get_face_embedding(faces[0], crop)
                if emb:
                    obj.embedding = face_service.merge_face_embeddings(
                        obj.embedding, emb, obj.reference_image_count
                    )
                    obj.reference_image_count += 1
                    added += 1
            else:
                # Fall back to body/CNN embedding for person without visible face
                obj.embedding = recognition_service.compute_and_merge_embedding(
                    crop, obj.embedding, obj.reference_image_count
                )
                obj.reference_image_count += 1
                added += 1
        else:
            obj.embedding = recognition_service.compute_and_merge_embedding(
                crop, obj.embedding, obj.reference_image_count
            )
            obj.reference_image_count += 1
            added += 1

    if added > 0:
        session.add(obj)
        await session.commit()

    return {"added": added, "total_training_images": obj.reference_image_count}


@router.get("/live-tracking")
async def live_tracking(session: AsyncSession = Depends(get_session)):
    """Return real-time tracking data: which objects are on which cameras right now.

    Combines:
    - Frigate in-progress events + very recent ended events (primary source)
    - ObjectTracker active tracks (enriched with names, bbox, confidence)
    - FrigateBridge presence tracking (named_object → camera mapping)

    Returns a list of live sightings with camera info and bounding boxes.
    """
    from services.object_tracker import object_tracker

    cameras_result = await session.execute(select(Camera).where(Camera.enabled == True))
    cameras = {c.id: c for c in cameras_result.scalars().all()}
    # Build reverse map: frigate camera name → (cam_id, cam_obj)
    frigate_to_cam = {f"camera_{c.id}": (c.id, c) for c in cameras.values()}

    sightings = []
    seen_frigate_ids = set()

    # 1. Primary source: Frigate active events (in-progress + recent ended)
    frigate_events = await frigate_bridge.get_active_frigate_events()
    for fev in frigate_events:
        cam_key = fev["frigate_camera"]
        if cam_key not in frigate_to_cam:
            continue
        cam_id, cam = frigate_to_cam[cam_key]
        seen_frigate_ids.add(fev["frigate_id"])
        sightings.append({
            "track_id": fev["frigate_id"],
            "camera_id": cam_id,
            "camera_name": cam.name,
            "class_name": fev["label"],
            "confidence": fev["score"],
            "bbox_norm": fev["bbox_norm"],
            "bbox": None,
            "named_object_name": fev.get("sub_label"),
            "in_progress": fev["in_progress"],
            "source": "frigate",
        })

    # 2. Supplement with internal ObjectTracker tracks (if running)
    for cam_id, cam in cameras.items():
        tracks = object_tracker.get_active_tracks(cam_id)
        for track_id, track in tracks.items():
            sightings.append({
                "track_id": str(track_id),
                "camera_id": cam_id,
                "camera_name": cam.name,
                "class_name": track.get("class_name", "object"),
                "confidence": track.get("confidence", 0),
                "bbox_norm": None,
                "bbox": track.get("bbox"),
                "named_object_name": track.get("named_object_name"),
                "in_progress": True,
                "source": "tracker",
            })

    # 3. Fill in presence-tracked named objects not already in sightings
    presence = frigate_bridge.get_current_presence()
    tracked_names = {s["named_object_name"] for s in sightings if s.get("named_object_name")}

    for name, cam_id in presence.items():
        if name not in tracked_names and cam_id in cameras:
            sightings.append({
                "track_id": None,
                "camera_id": cam_id,
                "camera_name": cameras[cam_id].name,
                "class_name": "object",
                "confidence": None,
                "bbox_norm": None,
                "bbox": None,
                "named_object_name": name,
                "in_progress": True,
                "source": "presence",
            })

    return sightings
