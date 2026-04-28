"""BanusNas — Role-based permissions.

Roles are a fixed hierarchy (admin > operator > viewer > guest). Each role
maps to a fixed bundle of feature permissions. Permission checks happen via
the `require_permission(...)` FastAPI dependency.

Why a fixed set: keeps the UI simple, avoids the operator-error footgun of
custom roles, and matches the stated requirements. If we ever need
arbitrary custom roles, the surface to swap is small (replace `ROLE_PERMISSIONS`
with a DB-backed lookup).
"""

from __future__ import annotations

from enum import Enum

from fastapi import Depends, HTTPException, status

from core.auth import get_current_user
from models.schemas import User, UserRole


class Permission(str, Enum):
    # Viewing
    view_live = "view_live"
    view_events = "view_events"
    view_recordings = "view_recordings"
    view_summaries = "view_summaries"
    view_search = "view_search"
    view_notifications = "view_notifications"

    # Operator-level — actions on detected content
    delete_events = "delete_events"
    train_objects = "train_objects"
    reanalyse_events = "reanalyse_events"
    manage_notifications = "manage_notifications"  # Own notification rules

    # Admin — system-wide configuration
    manage_cameras = "manage_cameras"
    manage_detection = "manage_detection"
    manage_credentials = "manage_credentials"
    manage_system = "manage_system"  # Settings, daily summaries config, ring config
    manage_users = "manage_users"
    view_audit_log = "view_audit_log"


# Bundles. Higher roles get the union of all lower-role permissions.
_VIEWER: set[Permission] = {
    Permission.view_live,
    Permission.view_events,
    Permission.view_recordings,
    Permission.view_summaries,
    Permission.view_search,
    Permission.view_notifications,
}

_OPERATOR: set[Permission] = _VIEWER | {
    Permission.delete_events,
    Permission.train_objects,
    Permission.reanalyse_events,
    Permission.manage_notifications,
}

_ADMIN: set[Permission] = _OPERATOR | {
    Permission.manage_cameras,
    Permission.manage_detection,
    Permission.manage_credentials,
    Permission.manage_system,
    Permission.manage_users,
    Permission.view_audit_log,
}

# Guest = nothing by default. Cameras can grant explicit per-camera viewer
# access in a future phase; for now guest is read-only nothing.
_GUEST: set[Permission] = set()

ROLE_PERMISSIONS: dict[str, set[Permission]] = {
    UserRole.admin.value: _ADMIN,
    UserRole.operator.value: _OPERATOR,
    UserRole.viewer.value: _VIEWER,
    UserRole.guest.value: _GUEST,
}


def user_role(user: User) -> str:
    """Resolve a user's effective role, falling back to legacy is_admin."""
    role = (user.role or "").strip()
    if role in ROLE_PERMISSIONS:
        return role
    # Legacy DBs where role wasn't backfilled yet.
    return UserRole.admin.value if user.is_admin else UserRole.viewer.value


def has_permission(user: User, perm: Permission) -> bool:
    return perm in ROLE_PERMISSIONS.get(user_role(user), set())


def permissions_for(user: User) -> list[str]:
    return sorted(p.value for p in ROLE_PERMISSIONS.get(user_role(user), set()))


def require_permission(perm: Permission):
    """FastAPI dependency factory.

    Usage:
        @router.delete("/events/{id}", dependencies=[Depends(require_permission(Permission.delete_events))])
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        if user.disabled:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")
        if not has_permission(user, perm):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {perm.value}",
            )
        return user

    return _check


def require_admin():
    """Shortcut for admin-only routes."""
    return require_permission(Permission.manage_users)
