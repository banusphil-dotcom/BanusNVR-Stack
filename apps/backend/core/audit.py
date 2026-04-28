"""BanusNas — Audit logging helper.

Captures security-relevant events (logins, user mutations, settings changes,
session revokes, etc.) into the `audit_logs` table.

Use sparingly — every call hits the DB. Reserve for actions an admin needs
to be able to investigate after the fact.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from models.schemas import AuditLog, User


def _client_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    # Honour reverse-proxy headers nginx sets in front of us.
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    if request.client:
        return request.client.host
    return None


async def audit(
    session: AsyncSession,
    *,
    action: str,
    actor: Optional[User] = None,
    actor_username: Optional[str] = None,
    target_type: Optional[str] = None,
    target_id: Optional[Any] = None,
    detail: Optional[dict] = None,
    request: Optional[Request] = None,
    commit: bool = False,
) -> None:
    """Insert an audit log row.

    Pass `commit=True` if you want this call to flush immediately instead of
    waiting for the surrounding transaction to commit (useful for failed
    login records that happen on a path that otherwise wouldn't commit).
    """
    entry = AuditLog(
        user_id=actor.id if actor else None,
        actor_username=actor_username or (actor.username if actor else None),
        action=action,
        target_type=target_type,
        target_id=str(target_id) if target_id is not None else None,
        detail=detail,
        ip_address=_client_ip(request),
        user_agent=(request.headers.get("user-agent") if request else None) or None,
    )
    session.add(entry)
    if commit:
        await session.commit()
