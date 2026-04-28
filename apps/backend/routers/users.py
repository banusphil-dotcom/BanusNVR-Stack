"""BanusNas — Admin user management API.

All endpoints require the `manage_users` permission (admin role). Provides
CRUD over user accounts plus the "kick device off" session revoke action.

Public registration is disabled (see routers.auth) — admins create accounts
here, optionally with `must_change_password=true` to force the new user to
pick their own password on first login.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.audit import audit
from core.auth import hash_password
from core.permissions import Permission, ROLE_PERMISSIONS, require_permission
from models.database import get_session
from models.schemas import AuditLog, NotificationRule, User, UserRole, UserSession
from schemas.api_schemas import UserResponse

router = APIRouter(prefix="/api/users", tags=["users"])


VALID_ROLES = {r.value for r in UserRole}


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    email: Optional[str] = None
    password: str = Field(min_length=8, max_length=128)
    role: str = UserRole.viewer.value
    must_change_password: bool = True


class UserUpdate(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    disabled: Optional[bool] = None
    must_change_password: Optional[bool] = None


class PasswordReset(BaseModel):
    new_password: str = Field(min_length=8, max_length=128)
    must_change_password: bool = True


def _serialize_user(u: User) -> dict:
    return {
        "id": u.id,
        "username": u.username,
        "email": u.email or "",
        "is_admin": u.is_admin,
        "role": u.role or (UserRole.admin.value if u.is_admin else UserRole.viewer.value),
        "theme": u.theme,
        "must_change_password": u.must_change_password,
        "disabled": u.disabled,
        "last_login_at": u.last_login_at,
        "created_at": u.created_at,
    }


@router.get("", response_model=list[UserResponse])
async def list_users(
    session: AsyncSession = Depends(get_session),
    _admin: User = Depends(require_permission(Permission.manage_users)),
):
    result = await session.execute(select(User).order_by(User.id))
    return [_serialize_user(u) for u in result.scalars().all()]


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_permission(Permission.manage_users)),
):
    if data.role not in VALID_ROLES:
        raise HTTPException(status_code=422, detail=f"Invalid role. Must be one of: {sorted(VALID_ROLES)}")

    existing = await session.execute(select(User).where(User.username == data.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(
        username=data.username,
        email=data.email or f"{data.username}@local",
        hashed_password=hash_password(data.password),
        is_admin=(data.role == UserRole.admin.value),
        role=data.role,
        must_change_password=data.must_change_password,
    )
    session.add(user)
    await session.flush()

    # Default notification rule, matching the bootstrap behaviour.
    session.add(NotificationRule(
        user_id=user.id,
        name="All Events",
        object_types=[],
        named_object_ids=[],
        camera_ids=[],
        channels={"push": True, "email": False},
        debounce_seconds=300,
        enabled=True,
    ))

    await audit(session, action="user.created", actor=admin, target_type="user", target_id=user.id,
                detail={"username": user.username, "role": user.role}, request=request)
    await session.commit()
    await session.refresh(user)
    return _serialize_user(user)


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UserUpdate,
    request: Request,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_permission(Permission.manage_users)),
):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    changes: dict = {}

    if data.email is not None and data.email != user.email:
        changes["email"] = {"from": user.email, "to": data.email}
        user.email = data.email

    if data.role is not None and data.role != user.role:
        if data.role not in VALID_ROLES:
            raise HTTPException(status_code=422, detail=f"Invalid role. Must be one of: {sorted(VALID_ROLES)}")
        # Prevent demoting the last admin.
        if user.role == UserRole.admin.value and data.role != UserRole.admin.value:
            admin_count = await session.scalar(
                select(func.count()).select_from(User).where(User.role == UserRole.admin.value)
            )
            if (admin_count or 0) <= 1:
                raise HTTPException(status_code=400, detail="Cannot demote the last remaining admin")
        changes["role"] = {"from": user.role, "to": data.role}
        user.role = data.role
        user.is_admin = (data.role == UserRole.admin.value)

    if data.disabled is not None and data.disabled != user.disabled:
        if data.disabled and user.role == UserRole.admin.value:
            admin_count = await session.scalar(
                select(func.count()).select_from(User).where(
                    User.role == UserRole.admin.value, User.disabled == False  # noqa: E712
                )
            )
            if (admin_count or 0) <= 1:
                raise HTTPException(status_code=400, detail="Cannot disable the last active admin")
        changes["disabled"] = {"from": user.disabled, "to": data.disabled}
        user.disabled = data.disabled
        # Disabling a user immediately revokes all their active sessions so
        # they're kicked off all devices, not just blocked from new logins.
        if data.disabled:
            await session.execute(
                UserSession.__table__.update()
                .where((UserSession.user_id == user.id) & (UserSession.revoked_at.is_(None)))
                .values(revoked_at=datetime.now(timezone.utc))
            )

    if data.must_change_password is not None and data.must_change_password != user.must_change_password:
        changes["must_change_password"] = {"from": user.must_change_password, "to": data.must_change_password}
        user.must_change_password = data.must_change_password

    if changes:
        await audit(session, action="user.updated", actor=admin, target_type="user", target_id=user.id,
                    detail={"username": user.username, "changes": changes}, request=request)

    await session.commit()
    await session.refresh(user)
    return _serialize_user(user)


@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    data: PasswordReset,
    request: Request,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_permission(Permission.manage_users)),
):
    """Admin-set password — also revokes all the user's active sessions."""
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(data.new_password)
    user.must_change_password = data.must_change_password
    user.failed_login_attempts = 0
    user.locked_until = None

    await session.execute(
        UserSession.__table__.update()
        .where((UserSession.user_id == user.id) & (UserSession.revoked_at.is_(None)))
        .values(revoked_at=datetime.now(timezone.utc))
    )

    await audit(session, action="user.password_reset", actor=admin, target_type="user", target_id=user.id,
                detail={"username": user.username, "must_change_password": data.must_change_password}, request=request)
    await session.commit()
    return {"message": "Password reset", "user_id": user.id}


@router.post("/{user_id}/unlock")
async def unlock_user(
    user_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_permission(Permission.manage_users)),
):
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.failed_login_attempts = 0
    user.locked_until = None
    await audit(session, action="user.unlocked", actor=admin, target_type="user", target_id=user.id,
                detail={"username": user.username}, request=request)
    await session.commit()
    return {"message": "Account unlocked"}


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_permission(Permission.manage_users)),
):
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.role == UserRole.admin.value:
        admin_count = await session.scalar(
            select(func.count()).select_from(User).where(User.role == UserRole.admin.value)
        )
        if (admin_count or 0) <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last remaining admin")

    username = user.username
    await session.delete(user)
    await audit(session, action="user.deleted", actor=admin, target_type="user", target_id=user_id,
                detail={"username": username}, request=request)
    await session.commit()
    return {"message": "User deleted"}


# ----- Sessions (admin view of all sessions) ------------------------------


@router.get("/{user_id}/sessions")
async def list_user_sessions(
    user_id: int,
    session: AsyncSession = Depends(get_session),
    _admin: User = Depends(require_permission(Permission.manage_users)),
):
    result = await session.execute(
        select(UserSession).where(UserSession.user_id == user_id).order_by(desc(UserSession.last_seen_at))
    )
    return [
        {
            "id": s.id,
            "user_agent": s.user_agent,
            "ip_address": s.ip_address,
            "created_at": s.created_at,
            "last_seen_at": s.last_seen_at,
            "revoked_at": s.revoked_at,
        }
        for s in result.scalars().all()
    ]


@router.delete("/{user_id}/sessions/{session_id}")
async def revoke_user_session(
    user_id: int,
    session_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
    admin: User = Depends(require_permission(Permission.manage_users)),
):
    result = await session.execute(select(UserSession).where(UserSession.id == session_id))
    sess = result.scalar_one_or_none()
    if not sess or sess.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.revoked_at is None:
        sess.revoked_at = datetime.now(timezone.utc)
        await audit(session, action="auth.session_revoked", actor=admin, target_type="session",
                    target_id=session_id, detail={"user_id": user_id}, request=request)
    await session.commit()
    return {"message": "Session revoked"}


# ----- Roles metadata for the UI ------------------------------------------


@router.get("/_meta/roles")
async def list_roles(_admin: User = Depends(require_permission(Permission.manage_users))):
    return [
        {"value": role, "permissions": sorted(p.value for p in perms)}
        for role, perms in ROLE_PERMISSIONS.items()
    ]
