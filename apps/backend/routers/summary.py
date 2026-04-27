"""BanusNas — Daily Summary API: retrieve and generate activity summaries."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from core.auth import get_current_user
from services.daily_summary import daily_summary_service

router = APIRouter(prefix="/api/summaries", tags=["summaries"])


@router.get("")
async def list_summaries(
    limit: int = Query(30, ge=1, le=100),
    _user=Depends(get_current_user),
):
    """List available summary dates."""
    dates = await daily_summary_service.get_available_dates(limit)
    return {"dates": dates}


@router.get("/{date_str}")
async def get_summary(
    date_str: str,
    summary_type: Optional[str] = None,
    _user=Depends(get_current_user),
):
    """Get summary for a specific date."""
    # Validate date format
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")

    summaries = await daily_summary_service.get_summary(date_str, summary_type)
    if not summaries:
        raise HTTPException(404, "No summary found for this date")
    return {"summaries": summaries}


@router.post("/{date_str}/generate")
async def generate_summary(
    date_str: str,
    summary_type: str = Query("morning", pattern="^(morning|evening)$"),
    _user=Depends(get_current_user),
):
    """Manually generate a summary for a date."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")

    data = await daily_summary_service.generate_summary(date_str, summary_type)
    return {"status": "ok", "data": data}


@router.post("/{date_str}/generate-deep")
async def generate_deep_summary(
    date_str: str,
    summary_type: str = Query("morning", pattern="^(morning|evening)$"),
    _user=Depends(get_current_user),
):
    """Re-generate summary narrative using the deep ML endpoint."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")

    data = await daily_summary_service.generate_deep_narrative(date_str, summary_type)
    return {"status": "ok", "data": data}
