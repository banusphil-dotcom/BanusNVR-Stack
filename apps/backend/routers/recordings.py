"""BanusNas — Recordings API: proxies Frigate NVR recording endpoints."""

import logging
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import Camera

logger = logging.getLogger(__name__)

RECORDINGS_ROOT = Path("/recordings")

router = APIRouter(prefix="/api/recordings", tags=["recordings"], dependencies=[Depends(get_current_user)])


async def _camera_stream_name(camera_id: int, session: AsyncSession) -> str:
    """Get Frigate camera name from database camera ID."""
    result = await session.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return f"camera_{camera.id}"


@router.get("/{camera_id}/playlist.m3u8")
async def get_playlist(
    camera_id: int,
    start: datetime = Query(..., description="Start time (ISO 8601)"),
    end: datetime = Query(..., description="End time (ISO 8601)"),
    session: AsyncSession = Depends(get_session),
):
    """Generate HLS playlist from Frigate recording segments."""
    stream_name = await _camera_stream_name(camera_id, session)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    # Get recording segments from Frigate API
    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
            resp = await client.get(
                f"/api/{stream_name}/recordings",
                params={"after": start_ts, "before": end_ts},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=404, detail="No recordings found for this time range")
            segments = resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Failed to get recording segments from Frigate: %s", e)
        raise HTTPException(status_code=404, detail="No recordings found for this time range")

    if not segments:
        raise HTTPException(status_code=404, detail="No recordings found for this time range")

    # Build HLS VOD playlist from local recording files
    max_duration = max(seg.get("duration", 10) for seg in segments)
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7",
        f"#EXT-X-TARGETDURATION:{int(max_duration) + 1}",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-MEDIA-SEQUENCE:0",
    ]
    for seg in segments:
        dur = seg.get("duration", 10.0)
        seg_id = seg.get("id", "")
        lines.append(f"#EXTINF:{dur:.3f},")
        lines.append(f"/api/recordings/{camera_id}/segment/{seg_id}")
    lines.append("#EXT-X-ENDLIST")

    return Response(
        content="\n".join(lines),
        media_type="application/vnd.apple.mpegurl",
    )


@router.get("/{camera_id}/segment/{segment_id}")
async def serve_segment(
    camera_id: int,
    segment_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Serve a recording segment as MPEG-TS (transmuxed from local MP4) for HLS."""
    import asyncio
    from datetime import timezone as tz_mod

    stream_name = await _camera_stream_name(camera_id, session)

    # Segment ID format: "1775833562.0-wnduym" → timestamp
    try:
        ts = float(segment_id.split("-")[0])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="Invalid segment ID")

    # Map timestamp → filesystem path (Frigate stores recordings in UTC)
    dt = datetime.fromtimestamp(ts, tz=tz_mod.utc)

    date_dir = dt.strftime("%Y-%m-%d")
    hour_dir = dt.strftime("%H")
    min_sec = dt.strftime("%M.%S")

    seg_dir = RECORDINGS_ROOT / date_dir / hour_dir / stream_name
    if not seg_dir.exists():
        raise HTTPException(status_code=404, detail="Recording directory not found")

    # Find exact or closest MP4 file
    target_file = seg_dir / f"{min_sec}.mp4"
    if not target_file.exists():
        mp4_files = sorted(seg_dir.glob("*.mp4"))
        if not mp4_files:
            raise HTTPException(status_code=404, detail="No recording files found")
        target_file = min(
            mp4_files,
            key=lambda f: abs(float(f.stem.replace(".", "", 1)) - float(min_sec.replace(".", "", 1))),
        )

    # Transmux MP4 → MPEG-TS (copy streams, no re-encoding — fast).
    # Stream ffmpeg's stdout directly to the client so hls.js can start
    # decoding immediately instead of waiting for the entire segment to
    # be transmuxed and buffered server-side. This is the difference
    # between "playback starts in 5s" and "playback is instant".
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(target_file),
        "-c", "copy", "-f", "mpegts",
        "-mpegts_flags", "+resend_headers",
        "pipe:1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _iter_chunks():
        try:
            assert proc.stdout is not None
            while True:
                chunk = await proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            try:
                await proc.wait()
            except Exception:
                pass

    return StreamingResponse(
        _iter_chunks(),
        media_type="video/mp2t",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@router.get("/{camera_id}/play.mp4")
async def play_recording(
    camera_id: int,
    start: datetime = Query(..., description="Start time (ISO 8601)"),
    end: datetime = Query(..., description="End time (ISO 8601)"),
    session: AsyncSession = Depends(get_session),
):
    """Stream a recording preview MP4 from Frigate for a time range."""
    stream_name = await _camera_stream_name(camera_id, session)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=60) as client:
            resp = await client.get(
                f"/api/{stream_name}/start/{start_ts}/end/{end_ts}/preview.mp4",
            )
            if resp.status_code == 200:
                return Response(
                    content=resp.content,
                    media_type="video/mp4",
                    headers={"Cache-Control": "public, max-age=3600"},
                )
    except Exception as e:
        logger.warning("Failed to get preview from Frigate: %s", e)

    raise HTTPException(status_code=404, detail="No recordings found")


@router.post("/{camera_id}/export")
async def export_clip(
    camera_id: int,
    start: datetime = Query(..., description="Start time (ISO 8601)"),
    end: datetime = Query(..., description="End time (ISO 8601)"),
    session: AsyncSession = Depends(get_session),
):
    """Export a time range as an MP4 clip via Frigate."""
    stream_name = await _camera_stream_name(camera_id, session)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=120) as client:
            resp = await client.get(
                f"/api/{stream_name}/recordings/{start_ts}/end/{end_ts}",
            )
            if resp.status_code == 200:
                filename = f"camera_{camera_id}_{start_ts}_{end_ts}.mp4"
                return Response(
                    content=resp.content,
                    media_type="video/mp4",
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    },
                )
    except Exception as e:
        logger.warning("Failed to export clip from Frigate: %s", e)

    raise HTTPException(status_code=404, detail="No recordings found or export failed")


@router.get("/{camera_id}/timeline")
async def get_timeline(
    camera_id: int,
    date: str = Query(..., description="Date (YYYY-MM-DD)"),
    session: AsyncSession = Depends(get_session),
):
    """Get recording timeline for a camera on a specific date from Frigate."""
    stream_name = await _camera_stream_name(camera_id, session)

    try:
        async with httpx.AsyncClient(base_url=settings.frigate_url, timeout=10) as client:
            resp = await client.get(f"/api/{stream_name}/recordings/summary")
            if resp.status_code == 200:
                data = resp.json()
                # Frigate returns list of {day, hours: [{hour, events, motion, objects, duration}]}
                # Frontend expects {camera_id, date, hours: [{hour, has_recording, segments}]}
                for day_data in data:
                    if day_data.get("day") == date:
                        hours = []
                        for h in day_data.get("hours", []):
                            hours.append({
                                "hour": h.get("hour", ""),
                                "has_recording": h.get("duration", 0) > 0,
                                "segments": h.get("motion", 0),
                            })
                        # Fill in missing hours (0-23) for a full day
                        existing_hours = {h["hour"] for h in hours}
                        for hh in range(24):
                            h_str = str(hh)
                            if h_str not in existing_hours:
                                hours.append({"hour": h_str, "has_recording": False, "segments": 0})
                        hours.sort(key=lambda x: int(x["hour"]))
                        return {"camera_id": camera_id, "date": date, "hours": hours}
                # No data for this date — return empty 24-hour timeline
                return {
                    "camera_id": camera_id,
                    "date": date,
                    "hours": [{"hour": str(h), "has_recording": False, "segments": 0} for h in range(24)],
                }
    except Exception as e:
        logger.warning("Failed to get timeline from Frigate: %s", e)

    return {
        "camera_id": camera_id,
        "date": date,
        "hours": [{"hour": str(h), "has_recording": False, "segments": 0} for h in range(24)],
    }
