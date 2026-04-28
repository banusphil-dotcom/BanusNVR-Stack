"""BanusNas — Audit log API (admin read-only)."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.permissions import Permission, require_permission
from models.database import get_session
from models.schemas import AuditLog, User

router = APIRouter(prefix="/api/audit-logs", tags=["audit"])


@router.get("")
async def list_audit_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    action: Optional[str] = Query(None, description="Substring filter on action"),
    user_id: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_session),
    _admin: User = Depends(require_permission(Permission.view_audit_log)),
):
    filters = []
    if action:
        filters.append(AuditLog.action.ilike(f"%{action}%"))
    if user_id is not None:
        filters.append(AuditLog.user_id == user_id)

    base = select(AuditLog)
    if filters:
        base = base.where(*filters)

    total_q = select(AuditLog.id)
    if filters:
        total_q = total_q.where(*filters)
    total = len((await session.execute(total_q)).all())

    rows = await session.execute(
        base.order_by(desc(AuditLog.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = [
        {
            "id": r.id,
            "user_id": r.user_id,
            "actor_username": r.actor_username,
            "action": r.action,
            "target_type": r.target_type,
            "target_id": r.target_id,
            "detail": r.detail,
            "ip_address": r.ip_address,
            "user_agent": r.user_agent,
            "created_at": r.created_at,
        }
        for r in rows.scalars().all()
    ]

    return {"items": items, "total": total, "page": page, "page_size": page_size}
