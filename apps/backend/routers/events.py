"""BanusNas — Events API: listing, filtering, snapshots, labeling, reanalysis."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from collections import OrderedDict

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user, get_user_from_token_param
from core.config import settings
from models.database import get_session
from models.schemas import Camera, Event, EventType, NamedObject, ObjectCategory
from schemas.api_schemas import EventLabel, EventResponse, EventsPage, EventGroupResponse, EventGroupsPage
from services.recognition_service import recognition_service
from services.face_service import face_service, FACE_EMBED_DIM
from services.narrative_generator import generate_narrative, generate_group_narrative, describe_snapshot_with_vision, describe_with_text_llm
from services.attribute_estimator import (
    estimate_person_attributes,
    compute_attribute_multiplier,
    merge_stable_attributes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])

_ANNOTATION_COLORS = [
    (80, 220, 120),
    (255, 180, 0),
    (80, 170, 255),
    (200, 120, 255),
    (255, 120, 120),
]


def _is_category_compatible(event_object_type: Optional[str], category: ObjectCategory) -> bool:
    if not event_object_type:
        return True
    mapping = {
        "person": {ObjectCategory.person},
        "cat": {ObjectCategory.pet},
        "dog": {ObjectCategory.pet},
        "bird": {ObjectCategory.pet},
        "car": {ObjectCategory.vehicle},
        "truck": {ObjectCategory.vehicle},
        "bus": {ObjectCategory.vehicle},
        "motorcycle": {ObjectCategory.vehicle},
        "bicycle": {ObjectCategory.vehicle},
        "boat": {ObjectCategory.vehicle},
    }
    allowed = mapping.get(event_object_type)
    if allowed is None:
        return category == ObjectCategory.other
    return category in allowed


def _resolve_pet_species(named_obj: NamedObject) -> Optional[str]:
    """Return the correct species label ('cat' or 'dog') for a pet profile.

    Checks explicit species attribute, breed, then returns None if unknown.
    Used to correct YOLO misclassification (e.g. white cat detected as 'dog').
    """
    attrs = named_obj.attributes or {}
    species = (attrs.get("species") or "").lower().strip()
    if species in ("cat", "dog"):
        return species
    breed = (attrs.get("breed") or "").lower().strip()
    if breed:
        from services.frigate_bridge import FrigateBridge
        if breed in FrigateBridge._CAT_BREEDS or any(b in breed for b in FrigateBridge._CAT_BREEDS):
            return "cat"
        if breed in FrigateBridge._DOG_BREEDS or any(b in breed for b in FrigateBridge._DOG_BREEDS):
            return "dog"
    return None


def _resolve_file(stored_path: str) -> Optional[Path]:
    """Resolve a stored file path, checking hot storage fallback.

    Files may be on SSD (hot) or HDD (cold). The DB path may reference either.
    """
    p = Path(stored_path)
    if p.exists():
        return p
    # Try alternate location: hot ↔ cold
    if settings.hot_storage_path:
        hot_snap = Path(settings.hot_storage_path) / "snapshots"
        cold_snap = Path(settings.snapshots_path)
        s = str(stored_path)
        if s.startswith(str(hot_snap)):
            alt = Path(s.replace(str(hot_snap), str(cold_snap), 1))
            if alt.exists():
                return alt
        elif s.startswith(str(cold_snap)):
            alt = Path(s.replace(str(cold_snap), str(hot_snap), 1))
            if alt.exists():
                return alt
    return None


def _build_event_annotations(event: Event, fallback_name: Optional[str]) -> list[dict]:
    meta = event.metadata_extra or {}
    annotations = meta.get("annotations") or []
    if annotations:
        return annotations
    if event.bbox:
        return [{
            "name": fallback_name or event.object_type or event.event_type.value,
            "class_name": event.object_type or "object",
            "bbox": event.bbox,
            "confidence": event.confidence,
            "primary": True,
        }]
    return []


def _scale_bbox_to_image(bbox: dict, image_w: int, image_h: int, detect_res: list | None) -> dict:
    """Scale bbox from detect coordinates to actual image coordinates."""
    if not detect_res or len(detect_res) < 2:
        return bbox
    det_w, det_h = detect_res[0], detect_res[1]
    if det_w <= 0 or det_h <= 0:
        return bbox
    # Only scale if there's a meaningful difference (>5%)
    sx = image_w / det_w
    sy = image_h / det_h
    if abs(sx - 1.0) < 0.05 and abs(sy - 1.0) < 0.05:
        return bbox
    return {
        "x1": int(bbox.get("x1", 0) * sx),
        "y1": int(bbox.get("y1", 0) * sy),
        "x2": int(bbox.get("x2", 0) * sx),
        "y2": int(bbox.get("y2", 0) * sy),
    }


def _clean_snapshot_path(snapshot_path: str) -> Optional[Path]:
    original = Path(snapshot_path)
    clean_candidate = original.with_name(f"{original.stem}_clean{original.suffix}")
    return _resolve_file(str(clean_candidate))


def _annotate_snapshot_image(image: np.ndarray, annotations: list[dict]) -> bytes:
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]
    for index, ann in enumerate(annotations):
        bbox = ann.get("bbox") or {}
        try:
            x1 = int(bbox.get("x1", bbox.get("0", 0)))
            y1 = int(bbox.get("y1", bbox.get("1", 0)))
            x2 = int(bbox.get("x2", bbox.get("2", 0)))
            y2 = int(bbox.get("y2", bbox.get("3", 0)))
        except (TypeError, ValueError):
            continue
        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _ANNOTATION_COLORS[index % len(_ANNOTATION_COLORS)]
        label = str(ann.get("name") or ann.get("class_name") or "object")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_top = max(0, y1 - text_h - baseline - 8)
        label_bottom = label_top + text_h + baseline + 8
        label_right = min(annotated.shape[1] - 1, x1 + text_w + 12)
        cv2.rectangle(annotated, (x1, label_top), (label_right, label_bottom), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 6, label_bottom - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
        )

    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to render snapshot")
    return encoded.tobytes()


@router.get("", response_model=EventsPage)
async def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    camera_id: Optional[int] = None,
    event_type: Optional[str] = None,
    object_type: Optional[str] = None,
    named_object_id: Optional[int] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    query = select(Event)
    count_query = select(func.count()).select_from(Event)

    filters = []
    if camera_id:
        filters.append(Event.camera_id == camera_id)
    if event_type:
        filters.append(Event.event_type == EventType(event_type))
    if object_type:
        if object_type.lower() == "pet":
            filters.append(Event.object_type.in_(["cat", "dog", "bird"]))
        else:
            filters.append(Event.object_type == object_type)
    if named_object_id:
        filters.append(Event.named_object_id == named_object_id)
    if from_date:
        filters.append(Event.started_at >= from_date)
    if to_date:
        filters.append(Event.started_at <= to_date)

    if filters:
        query = query.where(and_(*filters))
        count_query = count_query.where(and_(*filters))

    total = await session.scalar(count_query) or 0

    result = await session.execute(
        query.order_by(desc(Event.started_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    events = result.scalars().all()

    # Enrich with camera names and named object names
    items = []
    for ev in events:
        resp = EventResponse.model_validate(ev)

        # Get camera name
        cam_result = await session.execute(select(Camera.name).where(Camera.id == ev.camera_id))
        cam_name = cam_result.scalar_one_or_none()
        resp.camera_name = cam_name

        # Get named object name
        if ev.named_object_id:
            no_result = await session.execute(
                select(NamedObject.name).where(NamedObject.id == ev.named_object_id)
            )
            resp.named_object_name = no_result.scalar_one_or_none()
        resp.annotations = _build_event_annotations(ev, resp.named_object_name)

        # Convert paths to API URLs (always provide URLs — endpoint has Frigate fallback)
        resp.snapshot_path = f"/api/events/{ev.id}/snapshot"
        resp.thumbnail_path = f"/api/events/{ev.id}/thumbnail"

        # Duration and GIF from single-event lifecycle
        if ev.ended_at and ev.started_at:
            resp.duration = (ev.ended_at - ev.started_at).total_seconds()
        meta = ev.metadata_extra or {}
        if meta.get("gif_path"):
            resp.gif_url = f"/api/events/{ev.id}/gif"
        if meta.get("narrative"):
            resp.narrative = meta["narrative"]

        items.append(resp)

    return EventsPage(items=items, total=total, page=page, page_size=page_size)


# ── Group window constant (must match frigate_bridge.GROUP_WINDOW_S) ──
_GROUP_WINDOW_S = 60

# IoU threshold: unknowns with bbox overlap above this are considered the same object
_UNKNOWN_IOU_THRESHOLD = 0.40


def _bbox_iou(a: dict | None, b: dict | None) -> float:
    """Compute Intersection-over-Union between two {x1,y1,x2,y2} bboxes."""
    if not a or not b:
        return 0.0
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, a["x2"] - a["x1"]) * max(0, a["y2"] - a["y1"])
    area_b = max(0, b["x2"] - b["x1"]) * max(0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_group(events: list[Event]) -> list[Event]:
    """Deduplicate events within a group.

    Rules:
    - Named objects: keep only the highest-confidence event per named_object_id.
    - Unknowns (no named_object_id): keep only if bbox IoU with all already-kept
      unknowns is below threshold (i.e. spatially distinct objects).
    """
    # Sort by confidence descending so we keep the best first
    ranked = sorted(events, key=lambda e: (e.confidence or 0), reverse=True)

    seen_named: dict[int, Event] = {}  # named_object_id → best event
    kept_unknowns: list[Event] = []
    result: list[Event] = []

    for ev in ranked:
        if ev.named_object_id:
            if ev.named_object_id in seen_named:
                continue  # duplicate named object — skip
            seen_named[ev.named_object_id] = ev
            result.append(ev)
        else:
            # Unknown: check bbox overlap with already-kept unknowns
            dominated = False
            for existing in kept_unknowns:
                if _bbox_iou(ev.bbox, existing.bbox) > _UNKNOWN_IOU_THRESHOLD:
                    dominated = True
                    break
            if not dominated:
                kept_unknowns.append(ev)
                result.append(ev)

    # Re-sort by started_at to maintain chronological order
    result.sort(key=lambda e: e.started_at)
    return result


async def _enrich_event_response(ev: Event, session: AsyncSession) -> EventResponse:
    """Build an EventResponse from an Event ORM instance with all enrichments."""
    resp = EventResponse.model_validate(ev)
    cam_result = await session.execute(select(Camera.name).where(Camera.id == ev.camera_id))
    resp.camera_name = cam_result.scalar_one_or_none()
    if ev.named_object_id:
        no_result = await session.execute(
            select(NamedObject.name).where(NamedObject.id == ev.named_object_id)
        )
        resp.named_object_name = no_result.scalar_one_or_none()
    resp.annotations = _build_event_annotations(ev, resp.named_object_name)
    resp.snapshot_path = f"/api/events/{ev.id}/snapshot"
    resp.thumbnail_path = f"/api/events/{ev.id}/thumbnail"
    if ev.ended_at and ev.started_at:
        resp.duration = (ev.ended_at - ev.started_at).total_seconds()
    meta = ev.metadata_extra or {}
    if meta.get("gif_path"):
        resp.gif_url = f"/api/events/{ev.id}/gif"
    if meta.get("narrative"):
        resp.narrative = meta["narrative"]
    return resp


# Cross-camera session merging — kept tight so a recognised person/pet does not
# get merged into an "all night" run-on event. Hard cap on absolute session
# duration prevents long chains of merges from spanning hours.
_ACTIVITY_SESSION_WINDOW_S = 180.0  # 3 min idle gap before splitting cross-camera sessions
_MAX_SESSION_DURATION_S = 900.0     # 15 min absolute cap on a merged session


@router.get("/grouped", response_model=EventGroupsPage)
async def list_events_grouped(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    object_type: Optional[str] = Query(None),
    camera_id: Optional[int] = Query(None),
    recognised: Optional[bool] = Query(None, description="true=only named, false=only unknown"),
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """List events grouped into activity sessions.

    Phase 1: Group events by camera + time proximity (or group_key).
    Phase 2: Merge groups that share a named person/pet within a time window
             across different cameras → multi-room activity sessions.
    This reduces feed noise and gives a cross-room picture of activity.
    """
    # Build filters
    filters = []
    if object_type:
        # `pet` is a UI synonym that matches any cat/dog/bird detection so the
        # user doesn't have to care whether Frigate misclassified a cat as a
        # dog (which it regularly does on our cameras).
        if object_type.lower() == "pet":
            filters.append(Event.object_type.in_(["cat", "dog", "bird"]))
        else:
            filters.append(Event.object_type == object_type)
    if camera_id:
        filters.append(Event.camera_id == camera_id)
    if recognised is True:
        filters.append(Event.named_object_id.is_not(None))
    elif recognised is False:
        filters.append(Event.named_object_id.is_(None))

    # Fetch recent events (enough to form page_size groups + buffer)
    fetch_limit = page_size * 10
    query = select(Event).order_by(desc(Event.started_at)).limit(fetch_limit)
    if filters:
        query = query.where(and_(*filters))
    result = await session.execute(query)
    events = result.scalars().all()

    # ── Phase 1: Per-camera time-proximity grouping ──
    groups: OrderedDict[str, list[Event]] = OrderedDict()
    auto_group_latest: dict[int, tuple[str, float]] = {}

    for ev in events:
        if ev.group_key:
            groups.setdefault(ev.group_key, []).append(ev)
        else:
            ts = ev.started_at.timestamp() if ev.started_at else 0
            existing = auto_group_latest.get(ev.camera_id)
            if existing and abs(ts - existing[1]) < _GROUP_WINDOW_S:
                gk = existing[0]
                auto_group_latest[ev.camera_id] = (gk, ts)
            else:
                gk = f"auto_cam{ev.camera_id}_{ev.id}"
                auto_group_latest[ev.camera_id] = (gk, ts)
            groups.setdefault(gk, []).append(ev)

    # ── No cross-camera Phase 2 merging ──
    # The previous "activity session" merge collapsed every event sharing a
    # named_object_id across the page into one mega-group, producing a single
    # 10h+ card on the UI. Worse, it walked groups in DESC order but computed
    # windows as if ASC, so reassigning an unknown to a person dragged the
    # event into whatever long-running session was at the top of the page.
    # The frontend only needs the per-camera group_key buckets from Phase 1.
    merged_groups: OrderedDict[str, list[Event]] = groups

    # Count total groups (approximate)
    count_query = select(func.count()).select_from(Event)
    if filters:
        count_query = count_query.where(and_(*filters))
    total_events = await session.scalar(count_query) or 0
    events_in_batch = len(events)
    groups_in_batch = len(merged_groups)
    if events_in_batch > 0 and groups_in_batch > 0:
        ratio = events_in_batch / groups_in_batch
        total_groups = max(groups_in_batch, int(total_events / ratio))
    else:
        total_groups = 0

    # Paginate groups
    group_list = list(merged_groups.items())
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_groups = group_list[start_idx:end_idx]

    # Build response
    group_responses = []
    cam_name_cache: dict[int, str] = {}
    no_name_cache: dict[int, str] = {}

    async def _resolve_cam_name(cid: int) -> str:
        if cid not in cam_name_cache:
            r = await session.execute(select(Camera.name).where(Camera.id == cid))
            cam_name_cache[cid] = r.scalar_one_or_none() or f"Camera {cid}"
        return cam_name_cache[cid]

    async def _resolve_no_name(noid: int) -> str:
        if noid not in no_name_cache:
            r = await session.execute(select(NamedObject.name).where(NamedObject.id == noid))
            no_name_cache[noid] = r.scalar_one_or_none() or ""
        return no_name_cache[noid]

    for gk, group_events in page_groups:
        # Sort events within group by started_at
        group_events.sort(key=lambda e: e.started_at)
        # Deduplicate: one event per named object, distinct unknowns only
        group_events = _deduplicate_group(group_events)

        # Collect all cameras in order (for multi-room display)
        camera_id_order: list[int] = []
        seen_cams: set[int] = set()
        for ev in group_events:
            # Check metadata_extra.cameras for cross-camera consolidated events
            meta = ev.metadata_extra or {}
            ev_cameras = meta.get("cameras", [ev.camera_id])
            for cid in ev_cameras:
                if cid not in seen_cams:
                    seen_cams.add(cid)
                    camera_id_order.append(cid)

        # Build enriched event responses
        enriched_events: list[EventResponse] = []
        names: list[str] = []
        obj_types: list[str] = []
        group_camera_id = group_events[0].camera_id

        for ev in group_events:
            resp = EventResponse.model_validate(ev)
            resp.camera_name = await _resolve_cam_name(ev.camera_id)

            if ev.named_object_id:
                resp.named_object_name = await _resolve_no_name(ev.named_object_id)

            resp.annotations = _build_event_annotations(ev, resp.named_object_name)
            resp.snapshot_path = f"/api/events/{ev.id}/snapshot"
            resp.thumbnail_path = f"/api/events/{ev.id}/thumbnail"
            if ev.ended_at and ev.started_at:
                resp.duration = (ev.ended_at - ev.started_at).total_seconds()
            meta = ev.metadata_extra or {}
            if meta.get("gif_path"):
                resp.gif_url = f"/api/events/{ev.id}/gif"
            if meta.get("narrative"):
                resp.narrative = meta["narrative"]

            enriched_events.append(resp)
            names.append(resp.named_object_name or "")
            obj_types.append(ev.object_type or "person")

        # Group time range
        started = group_events[0].started_at
        ended = group_events[-1].ended_at or group_events[-1].started_at
        duration = (ended - started).total_seconds() if ended and started else None

        # Resolve all camera names for multi-room display
        all_camera_names = [await _resolve_cam_name(cid) for cid in camera_id_order]
        primary_camera_name = await _resolve_cam_name(group_camera_id)

        # Generate group narrative (multi-camera aware)
        narrative = generate_group_narrative(
            names=names,
            object_types=obj_types,
            camera_name=primary_camera_name,
            camera_names=all_camera_names if len(all_camera_names) > 1 else None,
            started_at=started,
            ended_at=ended,
        )

        unique_names = list(dict.fromkeys(n for n in names if n))

        primary = max(group_events, key=lambda e: (
            1 if e.named_object_id else 0,
            e.confidence or 0,
        ))

        group_responses.append(EventGroupResponse(
            group_key=gk,
            camera_id=group_camera_id,
            camera_name=primary_camera_name,
            camera_names=all_camera_names,
            started_at=started,
            ended_at=ended,
            duration=duration,
            narrative=narrative,
            names=unique_names,
            object_count=len(group_events),
            primary_event_id=primary.id,
            events=enriched_events,
        ))

    return EventGroupsPage(
        groups=group_responses,
        total_groups=total_groups,
        page=page,
        page_size=page_size,
    )


@router.get("/camera-timeline")
async def get_camera_timeline(
    camera_id: int = Query(...),
    date: str = Query(..., description="Date YYYY-MM-DD"),
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """Get events for a camera on a date — used for timeline display."""
    start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    result = await session.execute(
        select(Event)
        .where(Event.camera_id == camera_id, Event.started_at >= start, Event.started_at < end)
        .order_by(Event.started_at)
    )
    events = result.scalars().all()

    items = []
    for ev in events:
        named_name = None
        if ev.named_object_id:
            no_result = await session.execute(
                select(NamedObject.name).where(NamedObject.id == ev.named_object_id)
            )
            named_name = no_result.scalar_one_or_none()

        meta = ev.metadata_extra or {}
        items.append({
            "id": ev.id,
            "time": ev.started_at.isoformat(),
            "event_type": ev.event_type.value,
            "object_type": ev.object_type,
            "confidence": ev.confidence,
            "named_object_id": ev.named_object_id,
            "named_object_name": named_name,
            "motion_score": meta.get("motion_score"),
            "thumbnail_url": f"/api/events/{ev.id}/thumbnail",
        })

    return items


@router.get("/presence-timeline")
async def get_presence_timeline(
    camera_id: int = Query(...),
    date: str = Query(..., description="Date YYYY-MM-DD"),
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """Get presence bars for a camera on a given date.

    Returns named objects grouped by category (person, pet, other) with
    time bars showing when they were present. Cross-camera consolidated
    events that visited this camera are included.
    """
    start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    # Get all recognized events that touch this camera on this date.
    # Cross-camera events store all visited cameras in metadata_extra.cameras.
    result = await session.execute(
        select(Event)
        .where(
            Event.named_object_id.isnot(None),
            Event.started_at < end,
            func.coalesce(Event.ended_at, Event.started_at) >= start,
        )
        .order_by(Event.started_at)
    )
    events = result.scalars().all()

    # Filter to events that involve this camera (direct or via cameras list)
    relevant = []
    for ev in events:
        meta = ev.metadata_extra or {}
        cameras = meta.get("cameras", [ev.camera_id])
        if camera_id in cameras or ev.camera_id == camera_id:
            relevant.append(ev)

    # Build lookup for named object names + categories
    oid_set = {ev.named_object_id for ev in relevant}
    if oid_set:
        no_result = await session.execute(
            select(NamedObject).where(NamedObject.id.in_(oid_set))
        )
        named_map = {no.id: no for no in no_result.scalars().all()}
    else:
        named_map = {}

    # Group events by named_object_id into presence bars
    from collections import defaultdict
    obj_bars: dict[int, list[dict]] = defaultdict(list)
    for ev in relevant:
        oid = ev.named_object_id
        ev_start = max(ev.started_at, start)
        ev_end = min(ev.ended_at or ev.started_at + timedelta(seconds=30), end)
        obj_bars[oid].append({
            "start": ev_start.isoformat(),
            "end": ev_end.isoformat(),
            "event_id": ev.id,
            "cameras": (ev.metadata_extra or {}).get("cameras", [ev.camera_id]),
        })

    # Build response grouped by category
    category_order = {"person": 0, "pet": 1, "vehicle": 2, "other": 3}
    rows = []
    for oid, bars in obj_bars.items():
        obj = named_map.get(oid)
        if not obj:
            continue
        cat = obj.category.value if hasattr(obj.category, "value") else str(obj.category)
        rows.append({
            "named_object_id": oid,
            "name": obj.name,
            "category": cat,
            "bars": bars,
        })

    rows.sort(key=lambda r: (category_order.get(r["category"], 9), r["name"]))
    return rows


@router.post("/reclassify")
async def reclassify_recent(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """Start background reclassification of recent person events.

    Poll GET /reclassify/status for progress.
    """
    global _reclassify_state
    if _reclassify_state["running"]:
        raise HTTPException(status_code=409, detail="Reclassification already in progress")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await session.execute(
        select(Event.id).where(
            and_(
                Event.object_type == "person",
                Event.started_at >= cutoff,
                Event.snapshot_path.isnot(None),
            )
        ).order_by(desc(Event.started_at))
    )
    event_ids = [row[0] for row in result.all()]

    if not event_ids:
        return {"status": "done", "total": 0}

    _reclassify_state.update(
        running=True, phase="running", total=len(event_ids), current=0,
        face_found=0, recognition_changed=0, attribute_vetoed=0,
        agent_rejected=0, attributes_learned=0, error=None, details=[],
    )

    asyncio.create_task(_run_reclassify(event_ids))

    return {"status": "started", "total": len(event_ids)}


@router.get("/reclassify/status")
async def reclassify_status(
    since: int = Query(0, ge=0, description="Return details starting from this index"),
    _user=Depends(get_current_user),
):
    """Poll reclassification progress. Use ?since=N to get only new detail entries."""
    out = {k: v for k, v in _reclassify_state.items() if k != "details"}
    out["details"] = _reclassify_state["details"][since:]
    out["details_total"] = len(_reclassify_state["details"])
    return out


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, session: AsyncSession = Depends(get_session), _user=Depends(get_current_user)):
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    resp = EventResponse.model_validate(event)

    cam_result = await session.execute(select(Camera.name).where(Camera.id == event.camera_id))
    resp.camera_name = cam_result.scalar_one_or_none()

    if event.named_object_id:
        no_result = await session.execute(
            select(NamedObject.name).where(NamedObject.id == event.named_object_id)
        )
        resp.named_object_name = no_result.scalar_one_or_none()
    resp.annotations = _build_event_annotations(event, resp.named_object_name)

    resp.snapshot_path = f"/api/events/{event.id}/snapshot"
    resp.thumbnail_path = f"/api/events/{event.id}/thumbnail"

    # Duration and GIF from single-event lifecycle
    if event.ended_at and event.started_at:
        resp.duration = (event.ended_at - event.started_at).total_seconds()
    meta = event.metadata_extra or {}
    if meta.get("gif_path"):
        resp.gif_url = f"/api/events/{event.id}/gif"

    return resp


@router.get("/{event_id}/snapshot")
async def get_event_snapshot(
    event_id: int,
    annotated: bool = Query(True, description="Overlay bounding boxes and labels"),
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_user_from_token_param),
):
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Try local file first
    if event.snapshot_path:
        path = _resolve_file(event.snapshot_path)
        if path:
            if not annotated:
                # Return clean snapshot (no bounding boxes)
                clean = _clean_snapshot_path(event.snapshot_path)
                return Response(content=(clean or path).read_bytes(), media_type="image/jpeg")

            annotations = _build_event_annotations(event, None)
            if not annotations:
                return Response(content=path.read_bytes(), media_type="image/jpeg")

            image_path = _clean_snapshot_path(event.snapshot_path) or path
            image = cv2.imread(str(image_path))
            if image is None:
                return Response(content=path.read_bytes(), media_type="image/jpeg")

            # Scale bboxes from detect coords to actual image dims if needed
            detect_res = (event.metadata_extra or {}).get("detect_resolution")
            img_h, img_w = image.shape[:2]
            scaled_annotations = []
            for ann in annotations:
                ann_copy = dict(ann)
                if ann_copy.get("bbox"):
                    ann_copy["bbox"] = _scale_bbox_to_image(ann_copy["bbox"], img_w, img_h, detect_res)
                scaled_annotations.append(ann_copy)

            return Response(content=_annotate_snapshot_image(image, scaled_annotations), media_type="image/jpeg")

    # Fallback: fetch from Frigate by frigate_id
    meta = event.metadata_extra or {}
    frigate_id = meta.get("frigate_id")
    if frigate_id:
        try:
            params = {}
            if not annotated:
                params["bbox"] = "0"
            async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
                resp = await client.get(f"/api/events/{frigate_id}/snapshot.jpg", params=params)
                if resp.status_code == 200:
                    return Response(content=resp.content, media_type="image/jpeg")
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Snapshot not found")


@router.get("/{event_id}/thumbnail")
async def get_event_thumbnail(
    event_id: int,
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_user_from_token_param),
):
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Try local thumbnail file
    if event.thumbnail_path:
        path = _resolve_file(event.thumbnail_path)
        if path:
            return Response(content=path.read_bytes(), media_type="image/jpeg")

    # No local thumbnail — try on-the-fly crop from local snapshot + bbox
    if event.snapshot_path and event.bbox:
        snap_path = _resolve_file(event.snapshot_path)
        if snap_path:
            image = cv2.imread(str(snap_path))
            if image is not None:
                img_h, img_w = image.shape[:2]
                detect_res = (event.metadata_extra or {}).get("detect_resolution")
                scaled = _scale_bbox_to_image(event.bbox, img_w, img_h, detect_res)
                try:
                    x1 = max(0, int(scaled.get("x1", 0)))
                    y1 = max(0, int(scaled.get("y1", 0)))
                    x2 = min(img_w, int(scaled.get("x2", 0)))
                    y2 = min(img_h, int(scaled.get("y2", 0)))
                except (TypeError, ValueError):
                    x1, y1, x2, y2 = 0, 0, 0, 0
                if x2 > x1 and y2 > y1:
                    bw, bh = x2 - x1, y2 - y1
                    obj_type = event.object_type or ""
                    if obj_type == "person":
                        pt, pb, pl, pr = int(bh * 0.25), int(bh * 0.10), int(bw * 0.10), int(bw * 0.10)
                    else:
                        pt = pb = int(bh * 0.15)
                        pl = pr = int(bw * 0.15)
                    cx1, cy1 = max(0, x1 - pl), max(0, y1 - pt)
                    cx2, cy2 = min(img_w, x2 + pr), min(img_h, y2 + pb)
                    crop = image[cy1:cy2, cx1:cx2]
                    if crop.shape[0] >= 40 and crop.shape[1] >= 40:
                        # Resize to max 300px width for thumbnail
                        ch, cw = crop.shape[:2]
                        if cw > 300:
                            scale = 300 / cw
                            crop = cv2.resize(crop, (300, int(ch * scale)),
                                              interpolation=cv2.INTER_AREA)
                        ok, encoded = cv2.imencode(".jpg", crop,
                                                   [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if ok:
                            return Response(content=encoded.tobytes(),
                                            media_type="image/jpeg")

    # Try local snapshot as last resort (full-frame, no crop)
    if event.snapshot_path:
        path = _resolve_file(event.snapshot_path)
        if path:
            return Response(content=path.read_bytes(), media_type="image/jpeg")

    # Fallback: fetch from Frigate by frigate_id
    meta = event.metadata_extra or {}
    frigate_id = meta.get("frigate_id")
    if frigate_id:
        try:
            async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
                resp = await client.get(
                    f"/api/events/{frigate_id}/thumbnail.jpg",
                )
                if resp.status_code == 200:
                    return Response(content=resp.content, media_type="image/jpeg")
                # Try snapshot if thumbnail not available
                resp = await client.get(
                    f"/api/events/{frigate_id}/snapshot.jpg",
                    params={"quality": 70, "h": 200},
                )
                if resp.status_code == 200:
                    return Response(content=resp.content, media_type="image/jpeg")
        except Exception:
            pass

    raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.get("/{event_id}/crop")
async def get_event_crop(
    event_id: int,
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_user_from_token_param),
):
    """Return a cropped image of just the detected object with padding."""
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    bbox = event.bbox
    if not bbox:
        # No bbox — fall back to thumbnail behaviour
        return await get_event_thumbnail(event_id, session, _user)

    # Load snapshot image (prefer clean version without prior annotations)
    image = None
    if event.snapshot_path:
        path = _resolve_file(event.snapshot_path)
        if path:
            src = _clean_snapshot_path(event.snapshot_path) or path
            image = cv2.imread(str(src))

    if image is None:
        meta = event.metadata_extra or {}
        frigate_id = meta.get("frigate_id")
        if frigate_id:
            try:
                async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
                    resp = await client.get(
                        f"/api/events/{frigate_id}/snapshot.jpg", params={"bbox": "0"}
                    )
                    if resp.status_code == 200:
                        arr = np.frombuffer(resp.content, np.uint8)
                        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                pass

    if image is None:
        raise HTTPException(status_code=404, detail="Snapshot not found for cropping")

    h, w = image.shape[:2]

    # Scale bbox from detect coordinates to actual image coordinates if needed
    detect_res = (event.metadata_extra or {}).get("detect_resolution")
    scaled = _scale_bbox_to_image(bbox, w, h, detect_res)

    try:
        x1 = int(scaled.get("x1", scaled.get("0", 0)))
        y1 = int(scaled.get("y1", scaled.get("1", 0)))
        x2 = int(scaled.get("x2", scaled.get("2", 0)))
        y2 = int(scaled.get("y2", scaled.get("3", 0)))
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="Invalid bounding box")

    if x2 <= x1 or y2 <= y1:
        return await get_event_thumbnail(event_id, session, _user)

    # Asymmetric padding for persons (more on top for head), uniform for others
    bw, bh = x2 - x1, y2 - y1
    obj_type = event.object_type or ""
    if obj_type == "person":
        pad_top = int(bh * 0.25)
        pad_bottom = int(bh * 0.10)
        pad_left = int(bw * 0.10)
        pad_right = int(bw * 0.10)
    else:
        pad_top = pad_bottom = int(bh * 0.15)
        pad_left = pad_right = int(bw * 0.15)
    cx1 = max(0, x1 - pad_left)
    cy1 = max(0, y1 - pad_top)
    cx2 = min(w, x2 + pad_right)
    cy2 = min(h, y2 + pad_bottom)

    crop = image[cy1:cy2, cx1:cx2]
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode crop")
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


@router.get("/{event_id}/gif")
async def get_event_gif(
    event_id: int,
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_user_from_token_param),
):
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    meta = event.metadata_extra or {}
    gif_path_str = meta.get("gif_path")
    if not gif_path_str:
        raise HTTPException(status_code=404, detail="No GIF for this event")

    path = _resolve_file(gif_path_str)
    if not path:
        raise HTTPException(status_code=404, detail="GIF file not found")

    return Response(content=path.read_bytes(), media_type="image/gif")


@router.post("/{event_id}/label")
async def label_event(event_id: int, data: EventLabel, session: AsyncSession = Depends(get_session), _user=Depends(get_current_user)):
    """Assign a named object to a detection event. Also trains the model with the event crop.

    When reassigning from one person to another, rebuilds the old person's
    embedding from their remaining assigned events so the model stays accurate.
    """
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Verify named object exists
    no_result = await session.execute(select(NamedObject).where(NamedObject.id == data.named_object_id))
    named_obj = no_result.scalar_one_or_none()
    if not named_obj:
        raise HTTPException(status_code=404, detail="Named object not found")

    if not _is_category_compatible(event.object_type, named_obj.category):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot assign {named_obj.category.value} object to event type '{event.object_type}'",
        )

    # Track old assignment for rebuild
    old_object_id = event.named_object_id
    is_reassign = old_object_id is not None and old_object_id != data.named_object_id

    # Update event
    event.named_object_id = data.named_object_id
    event.event_type = EventType.object_recognized

    # Correct species label for pets (e.g. YOLO 'dog' → 'cat' for known cats)
    if named_obj.category == ObjectCategory.pet and event.object_type in ("cat", "dog"):
        corrected = _resolve_pet_species(named_obj)
        if corrected and corrected != event.object_type:
            event.object_type = corrected

    session.add(event)

    # Auto-train: use bbox crop (not full frame) as training data for NEW object
    trained = False
    if event.thumbnail_path and Path(event.thumbnail_path).exists():
        from routers.training import _train_object_with_bytes, _load_event_crop
        img_bytes = _load_event_crop(event)
        if img_bytes:
            trained = await _train_object_with_bytes(named_obj, img_bytes)
        if trained:
            named_obj.reference_image_count += 1
            session.add(named_obj)

    await session.commit()

    # Rebuild old object's model in background (can take minutes for 100 events)
    if is_reassign:
        asyncio.create_task(_rebuild_object_embeddings_bg(old_object_id, event_id))

    return {"message": "Event labeled", "trained": trained, "old_model_rebuilt": is_reassign}


async def _rebuild_object_embeddings_bg(object_id: int, event_id: int):
    """Background task: rebuild a named object's embeddings from assigned events.

    For pets, uses the colour-gate (services/pet_color_gate.py) to drop crops
    whose observed fur-colour family is incompatible with the profile's
    ``attributes['color']`` — this prevents centroid contamination of the kind
    that produced the Frostie/Tangie cross-confusion.

    Runs with its own DB session so the label endpoint can return immediately.
    """
    from models.database import async_session
    from routers.training import _load_event_crop
    from services.pet_color_gate import (
        compute_colour_signal, colour_compatibility,
    )
    from services.ml_client import remote_embedding
    try:
        async with async_session() as session:
            obj = (await session.execute(select(NamedObject).where(NamedObject.id == object_id))).scalar_one_or_none()
            if not obj:
                return

            ev_result = await session.execute(
                select(Event)
                .where(and_(Event.named_object_id == obj.id, Event.thumbnail_path.isnot(None)))
                .order_by(desc(Event.started_at))
                .limit(50)
            )
            events = ev_result.scalars().all()

            obj.embedding = None
            obj.body_embedding = None
            obj.reference_image_count = 0
            is_person = obj.category == ObjectCategory.person
            is_pet = obj.category == ObjectCategory.pet
            profile_color = None
            if is_pet:
                attrs = obj.attributes or {}
                profile_color = attrs.get("color") or attrs.get("colour")

            colour_dropped = 0
            for ev in events:
                img_bytes = _load_event_crop(ev)
                if not img_bytes:
                    continue
                crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if crop is None:
                    continue

                # Colour gate (pets only): drop crops vetoed against profile colour.
                if is_pet and profile_color:
                    signal = compute_colour_signal(crop)
                    if colour_compatibility(profile_color, signal) <= 0.0:
                        colour_dropped += 1
                        continue

                if is_person and face_service.is_available:
                    faces = face_service.detect_faces(crop)
                    if faces:
                        emb, _ = face_service.compute_face_embedding(crop, faces[0].face_data)
                        if emb:
                            obj.embedding = face_service.merge_face_embeddings(
                                obj.embedding, emb, obj.reference_image_count
                            )
                    body_emb = recognition_service.compute_reid_embedding(crop)
                    if body_emb:
                        obj.body_embedding = recognition_service.merge_reid_embedding(
                            obj.body_embedding, body_emb, obj.reference_image_count
                        )
                else:
                    # Force remote 1280-d MobileNetV2 to match the live pipeline.
                    # Local fallback is only 192-d histogram, which is incompatible
                    # with the centroid the live `_recognize_pet` will compare against.
                    try:
                        emb = await remote_embedding(crop, model="cnn")
                    except Exception as e:
                        logger.warning("Remote embedding failed for ev %d: %s", ev.id, e)
                        emb = None
                    if emb:
                        ea = np.array(emb, dtype=np.float32)
                        if obj.embedding is None:
                            obj.embedding = ea.tolist()
                        else:
                            old = np.array(obj.embedding, dtype=np.float32)
                            if len(old) != len(ea):
                                obj.embedding = ea.tolist()
                                obj.reference_image_count = 0
                            else:
                                merged = (old * obj.reference_image_count + ea) / (obj.reference_image_count + 1)
                                merged = merged / max(np.linalg.norm(merged), 1e-8)
                                obj.embedding = merged.tolist()

                obj.reference_image_count += 1

            session.add(obj)
            await session.commit()
            logger.info("Reassign event %d: rebuilt %s model in background (refs=%d, colour_dropped=%d)",
                        event_id, obj.name, obj.reference_image_count, colour_dropped)
    except Exception:
        logger.exception("Background rebuild failed for object %d", object_id)


@router.get("/{event_id}/suggestions")
async def get_event_suggestions(
    event_id: int,
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """For unrecognized person events, compare the thumbnail against all known
    people and return ranked suggestions with confidence percentages."""
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    if not event.thumbnail_path or not Path(event.thumbnail_path).exists():
        return {"suggestions": []}

    # Use bbox crop to isolate the target person for matching
    from routers.training import _load_event_crop
    img_bytes = _load_event_crop(event)
    if not img_bytes:
        return {"suggestions": []}
    crop = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )
    if crop is None:
        return {"suggestions": []}

    # Get all named persons
    obj_result = await session.execute(
        select(NamedObject).where(NamedObject.category == ObjectCategory.person)
    )
    named_persons = obj_result.scalars().all()
    if not named_persons:
        return {"suggestions": []}

    # Compute embeddings from the event thumbnail
    face_emb = None
    body_emb = None
    if face_service.is_available:
        faces = face_service.detect_faces(crop)
        if faces:
            face_emb, _ = face_service.compute_face_embedding(crop, faces[0].face_data)
    if recognition_service.reid_available:
        body_emb = recognition_service.compute_reid_embedding(crop)

    if face_emb is None and body_emb is None:
        return {"suggestions": []}

    suggestions = []
    for person in named_persons:
        face_sim = None
        body_sim = None

        if face_emb and person.embedding:
            stored = np.array(person.embedding, dtype=np.float32)
            if len(stored) == FACE_EMBED_DIM:
                face_sim = float(face_service.cosine_similarity(face_emb, person.embedding))

        if body_emb and person.body_embedding:
            stored_body = np.array(person.body_embedding, dtype=np.float32)
            b = np.array(body_emb, dtype=np.float32)
            body_sim = float(np.dot(b, stored_body) / max(np.linalg.norm(b) * np.linalg.norm(stored_body), 1e-8))

        # Compute combined score (same weighting as scan)
        if face_sim is not None and body_sim is not None:
            if face_sim >= 0.40:
                score = face_sim * 0.65 + body_sim * 0.35
            elif face_sim < 0.30:
                score = face_sim * 0.3 + body_sim * 0.15
            else:
                score = face_sim * 0.5 + body_sim * 0.5
        elif face_sim is not None:
            score = face_sim
        elif body_sim is not None:
            score = body_sim * 0.80
        else:
            continue

        # Only include if there's a meaningful match
        if score >= 0.20:
            suggestions.append({
                "named_object_id": person.id,
                "name": person.name,
                "confidence": round(score * 100),
                "face_similarity": round(face_sim, 3) if face_sim else None,
                "body_similarity": round(body_sim, 3) if body_sim else None,
            })

    suggestions.sort(key=lambda s: s["confidence"], reverse=True)
    return {"suggestions": suggestions[:5]}


@router.post("/{event_id}/reanalyse")
async def reanalyse_event(
    event_id: int,
    object_type: Optional[str] = Query(None, description="Object type (person, cat, dog, car, etc.)"),
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """Re-run recognition on an event's snapshot/thumbnail.

    Optionally override object_type.  Uses the existing crop from
    the event thumbnail, or fetches a fresh snapshot from Frigate.
    """
    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    obj_type = object_type or event.object_type or "person"
    # The UI exposes a single "pet" reanalyse button (cat/dog/bird are
    # frequently mis-classified by Frigate, so we don't want users to have
    # to pick the right one). Map back to a concrete animal class for the
    # downstream pet recognition + attribute pipeline — prefer the event's
    # own object_type when it's already a valid animal class.
    if obj_type.lower() == "pet":
        original = (event.object_type or "").lower()
        obj_type = original if original in ("cat", "dog", "bird") else "cat"

    # Get crop image — prefer bbox crop from full snapshot to isolate the target
    crop = None
    from routers.training import _load_event_crop
    img_bytes = _load_event_crop(event)
    if img_bytes:
        crop = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if crop is None and event.thumbnail_path:
        thumb = _resolve_file(event.thumbnail_path)
        if thumb:
            crop = cv2.imdecode(np.frombuffer(thumb.read_bytes(), np.uint8), cv2.IMREAD_COLOR)

    # Fallback: fetch from Frigate
    if crop is None:
        meta = event.metadata_extra or {}
        frigate_id = meta.get("frigate_id")
        if frigate_id:
            try:
                async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
                    resp = await client.get(f"/api/events/{frigate_id}/thumbnail.jpg")
                    if resp.status_code == 200:
                        crop = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                pass

    if crop is None:
        raise HTTPException(status_code=422, detail="No image available for reanalysis")

    # Run recognition
    named_object_id = None
    named_object_name = None
    match_method = None

    if obj_type == "person":
        # Face recognition
        if face_service.is_available:
            faces = face_service.detect_faces(crop)
            if faces:
                emb, _ = face_service.compute_face_embedding(crop, faces[0].face_data)
                if emb:
                    known_result = await session.execute(
                        select(NamedObject).where(
                            NamedObject.category == ObjectCategory.person,
                            NamedObject.embedding.isnot(None),
                        )
                    )
                    known_persons = [(p.id, p.name, p.embedding) for p in known_result.scalars().all()]
                    if known_persons:
                        face_match = face_service.match_face(emb, known_persons)
                        if face_match:
                            cand_name, cand_id, raw_score, _ = face_match
                            if raw_score >= 0.28:
                                named_object_name, named_object_id = cand_name, cand_id
                                match_method = "face"

        # Body ReID fallback
        if named_object_id is None and recognition_service.reid_available:
            body_result = await session.execute(
                select(NamedObject).where(
                    NamedObject.category == ObjectCategory.person,
                    NamedObject.body_embedding.isnot(None),
                )
            )
            known_bodies = [(p.id, p.name, p.body_embedding) for p in body_result.scalars().all()]
            if known_bodies:
                body_match = await recognition_service.match_person_body(crop, known_bodies)
                if body_match and body_match.subject and body_match.confidence >= 0.40:
                    named_object_name = body_match.subject
                    named_object_id = int(body_match.subject_id)
                    match_method = "body"

    elif obj_type in ("cat", "dog"):
        pets_result = await session.execute(
            select(NamedObject).where(
                NamedObject.category == ObjectCategory.pet,
                NamedObject.embedding.isnot(None),
            )
        )
        known_pets = [(p.id, p.name, p.embedding) for p in pets_result.scalars().all()]
        if known_pets:
            rec_result = await recognition_service.match_pet(crop, known_pets)
            if rec_result and rec_result.subject:
                named_object_name = rec_result.subject
                named_object_id = int(rec_result.subject_id)
                match_method = "cnn"

    elif obj_type in ("car", "truck", "bus", "motorcycle", "bicycle", "boat"):
        veh_result = await session.execute(
            select(NamedObject).where(
                NamedObject.category == ObjectCategory.vehicle,
                NamedObject.embedding.isnot(None),
            )
        )
        known = [(p.id, p.name, p.embedding) for p in veh_result.scalars().all()]
        if known:
            rec_result = await recognition_service.match_pet(crop, known)
            if rec_result and rec_result.subject:
                named_object_name = rec_result.subject
                named_object_id = int(rec_result.subject_id)
                match_method = "cnn"

    # Get camera name for narrative
    cam_result = await session.execute(select(Camera.name).where(Camera.id == event.camera_id))
    camera_name = cam_result.scalar_one_or_none() or f"camera_{event.camera_id}"

    # Vision-based narrative: use full-frame snapshot (not crop)
    vision_frame = None
    if event.snapshot_path:
        snap_file = _resolve_file(event.snapshot_path)
        if snap_file:
            vision_frame = cv2.imdecode(np.frombuffer(snap_file.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if vision_frame is None and crop is not None:
        vision_frame = crop  # fallback to crop if no full frame

    narrative = None
    narrative_source = "factual"
    if vision_frame is not None and obj_type in ("person", "cat", "dog"):
        narrative = await describe_snapshot_with_vision(
            vision_frame,
            camera_name=camera_name,
            object_type=obj_type,
            named_object_name=named_object_name,
            timestamp=event.started_at,
        )
        if narrative:
            narrative_source = "vision"
    if not narrative:
        narrative = await describe_with_text_llm(
            camera_name=camera_name,
            object_type=obj_type,
            named_object_name=named_object_name,
            timestamp=event.started_at,
        )
        if narrative:
            narrative_source = "llm"
    if not narrative:
        narrative = generate_narrative(
            named_object_name=named_object_name,
            object_type=obj_type,
            camera_name=camera_name,
            timestamp=event.started_at,
        )
        narrative_source = "factual"

    # Update event
    event_type = EventType.object_recognized if named_object_id else EventType.object_detected
    event.object_type = obj_type
    event.event_type = event_type
    event.named_object_id = named_object_id

    # Correct species label for matched pets
    if named_object_id and obj_type in ("cat", "dog"):
        matched_obj = await session.get(NamedObject, named_object_id)
        if matched_obj and matched_obj.category == ObjectCategory.pet:
            corrected = _resolve_pet_species(matched_obj)
            if corrected:
                event.object_type = corrected

    meta = dict(event.metadata_extra or {})
    meta["reanalysed"] = True
    meta["narrative"] = narrative
    meta["narrative_source"] = narrative_source
    if match_method:
        meta["match_method"] = match_method
    event.metadata_extra = meta
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(event, "metadata_extra")
    await session.commit()

    return {
        "event_id": event_id,
        "object_type": obj_type,
        "named_object": named_object_name,
        "match_method": match_method,
        "narrative": narrative,
        "updated": True,
    }


# ──────────── Reclassify background task ────────────

_reclassify_state: dict = {
    "running": False,
    "phase": "idle",
    "total": 0,
    "current": 0,
    "face_found": 0,
    "recognition_changed": 0,
    "error": None,
    "details": [],
}


async def _run_reclassify(event_ids: list[int]):
    """Background coroutine: reclassify person events using face + body recognition."""
    global _reclassify_state
    from models.database import async_session
    try:
        async with async_session() as session:
            result = await session.execute(
                select(Event).where(Event.id.in_(event_ids))
            )
            events = result.scalars().all()
            _reclassify_state["total"] = len(events)

            # Pre-load known persons
            person_q = await session.execute(
                select(NamedObject).where(NamedObject.category == ObjectCategory.person)
            )
            all_persons = person_q.scalars().all()
            known_faces = [(p.id, p.name, p.embedding) for p in all_persons if p.embedding]
            known_bodies = [(p.id, p.name, p.body_embedding) for p in all_persons if p.body_embedding]

            for i, ev in enumerate(events):
                _reclassify_state["current"] = i + 1

                crop = None
                if ev.thumbnail_path:
                    thumb = Path(ev.thumbnail_path)
                    if thumb.exists():
                        crop = cv2.imdecode(np.frombuffer(thumb.read_bytes(), np.uint8), cv2.IMREAD_COLOR)

                if crop is None:
                    continue

                new_match_id = None
                new_match_name = None
                match_method = None

                # Face detection + recognition
                if face_service.is_available:
                    faces = face_service.detect_faces(crop)
                    if faces:
                        _reclassify_state["face_found"] += 1
                        emb, _ = face_service.compute_face_embedding(crop, faces[0].face_data)
                        if emb and known_faces:
                            match = face_service.match_face(emb, known_faces)
                            if match:
                                cand_name, cand_id, raw_score, _ = match
                                if raw_score >= 0.28:
                                    new_match_name, new_match_id = cand_name, cand_id
                                    match_method = "face"

                # Body ReID fallback
                if new_match_id is None and recognition_service.reid_available and known_bodies:
                    body_match = await recognition_service.match_person_body(crop, known_bodies, threshold=0.30)
                    if body_match and body_match.subject and body_match.confidence >= 0.40:
                        new_match_name = body_match.subject
                        new_match_id = int(body_match.subject_id)
                        match_method = "body"

                old_id = ev.named_object_id
                changed = False
                if new_match_id is not None and new_match_id != old_id:
                    ev.named_object_id = new_match_id
                    ev.event_type = EventType.object_recognized
                    _reclassify_state["recognition_changed"] += 1
                    changed = True

                detail: dict = {"event_id": ev.id}
                if new_match_name:
                    detail["match"] = new_match_name
                    detail["method"] = match_method
                    if changed:
                        old_obj = await session.get(NamedObject, old_id) if old_id else None
                        detail["was"] = old_obj.name if old_obj else "unknown"
                else:
                    detail["match"] = None
                _reclassify_state["details"].append(detail)

                if i % 5 == 0:
                    await asyncio.sleep(0)

            await session.commit()
            _reclassify_state["phase"] = "done"

    except Exception as e:
        logger.error("Reclassify background task error: %s", e)
        _reclassify_state.update(phase="error", error=str(e))
    finally:
        _reclassify_state["running"] = False


@router.post("/{event_id}/false-positive")
async def mark_false_positive(
    event_id: int,
    session: AsyncSession = Depends(get_session),
    _user=Depends(get_current_user),
):
    """Mark a person detection as 'not a person'. Stores the body embedding as a
    negative example so future similar detections are suppressed, then deletes
    the event.
    """
    from models.schemas import SystemSettings

    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # Extract body embedding from thumbnail for the negative example
    embedding_stored = False
    if event.thumbnail_path and Path(event.thumbnail_path).exists():
        crop = cv2.imdecode(
            np.frombuffer(Path(event.thumbnail_path).read_bytes(), np.uint8),
            cv2.IMREAD_COLOR,
        )
        if crop is not None and recognition_service.reid_available:
            body_emb = recognition_service.compute_reid_embedding(crop)
            if body_emb:
                fp_result = await session.execute(
                    select(SystemSettings).where(SystemSettings.key == "false_positive_embeddings")
                )
                fp_setting = fp_result.scalar_one_or_none()
                if fp_setting:
                    data = fp_setting.value or {}
                    embeddings = data.get("embeddings", [])
                    embeddings.append(body_emb if isinstance(body_emb, list) else list(body_emb))
                    if len(embeddings) > 200:
                        embeddings = embeddings[-200:]
                    fp_setting.value = {"embeddings": embeddings}
                else:
                    fp_setting = SystemSettings(
                        key="false_positive_embeddings",
                        value={"embeddings": [body_emb if isinstance(body_emb, list) else list(body_emb)]},
                    )
                    session.add(fp_setting)
                embedding_stored = True

    # Clean up files
    for path_str in [event.snapshot_path, event.thumbnail_path]:
        if path_str:
            Path(path_str).unlink(missing_ok=True)
    meta = event.metadata_extra or {}
    if meta.get("gif_path"):
        Path(meta["gif_path"]).unlink(missing_ok=True)

    await session.delete(event)
    await session.commit()

    return {"message": "Marked as false positive", "embedding_stored": embedding_stored}


@router.delete("/{event_id}", status_code=204)
async def delete_event(event_id: int, session: AsyncSession = Depends(get_session), _user=Depends(get_current_user)):
    from models.schemas import SystemSettings

    result = await session.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    # For person events: store false-positive embedding so similar detections
    # are suppressed in the future.
    if event.object_type == "person" and not event.named_object_id:
        if event.thumbnail_path and Path(event.thumbnail_path).exists():
            try:
                crop = cv2.imdecode(
                    np.frombuffer(Path(event.thumbnail_path).read_bytes(), np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if crop is not None and recognition_service.reid_available:
                    body_emb = recognition_service.compute_reid_embedding(crop)
                    if body_emb:
                        fp_result = await session.execute(
                            select(SystemSettings).where(SystemSettings.key == "false_positive_embeddings")
                        )
                        fp_setting = fp_result.scalar_one_or_none()
                        if fp_setting:
                            data = fp_setting.value or {}
                            embeddings = data.get("embeddings", [])
                            embeddings.append(body_emb if isinstance(body_emb, list) else list(body_emb))
                            if len(embeddings) > 200:
                                embeddings = embeddings[-200:]
                            fp_setting.value = {"embeddings": embeddings}
                        else:
                            fp_setting = SystemSettings(
                                key="false_positive_embeddings",
                                value={"embeddings": [body_emb if isinstance(body_emb, list) else list(body_emb)]},
                            )
                            session.add(fp_setting)
            except Exception:
                pass  # Best-effort — don't block deletion

    # Clean up files (snapshot, thumbnail, GIF)
    for path_str in [event.snapshot_path, event.thumbnail_path]:
        if path_str:
            Path(path_str).unlink(missing_ok=True)
    meta = event.metadata_extra or {}
    if meta.get("gif_path"):
        Path(meta["gif_path"]).unlink(missing_ok=True)

    await session.delete(event)
    await session.commit()

    return Response(status_code=204)
