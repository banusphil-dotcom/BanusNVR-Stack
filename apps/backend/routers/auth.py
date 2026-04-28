"""BanusNas — Auth API: login, register, refresh, profile, sessions."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.audit import audit, _client_ip
from core.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    is_locked_out,
    register_failed_login,
    register_successful_login,
    verify_password,
    generate_totp_secret,
    get_totp_uri,
    verify_totp,
)
# ----- TOTP 2FA endpoints -------------------------------------------------

class TOTPSetupResponse(BaseModel):
    secret: str
    uri: str

@router.post("/totp/setup", response_model=TOTPSetupResponse)
async def totp_setup(user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    """Begin TOTP setup: returns secret and provisioning URI."""
    if user.totp_enabled:
        raise HTTPException(400, "TOTP already enabled")
    secret = generate_totp_secret()
    user.totp_secret = secret
    session.add(user)
    await session.commit()
    return TOTPSetupResponse(secret=secret, uri=get_totp_uri(user.username, secret))


class TOTPVerifyRequest(BaseModel):
    token: str

@router.post("/totp/verify")
async def totp_verify(
    data: TOTPVerifyRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not user.totp_secret:
        raise HTTPException(400, "No TOTP secret set up")
    if not verify_totp(data.token, user.totp_secret):
        raise HTTPException(400, "Invalid TOTP code")
    user.totp_enabled = True
    session.add(user)
    await session.commit()
    return {"message": "TOTP enabled"}


@router.post("/totp/disable")
async def totp_disable(user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    user.totp_enabled = False
    user.totp_secret = None
    session.add(user)
    await session.commit()
    return {"message": "TOTP disabled"}
from core.permissions import permissions_for, user_role
from models.database import get_session
from models.schemas import NotificationRule, User, UserRole, UserSession
from schemas.api_schemas import (
    TokenRefresh,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


async def _create_session(session: AsyncSession, user: User, request: Request) -> UserSession:
    sess = UserSession(
        user_id=user.id,
        user_agent=(request.headers.get("user-agent") or "")[:500],
        ip_address=_client_ip(request),
    )
    session.add(sess)
    await session.flush()  # populate sess.id
    return sess


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    data: UserRegister,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """First-user-becomes-admin bootstrap.

    Once any user exists, this endpoint refuses public registration. Admins
    must use POST /api/users to create additional accounts.
    """
    user_count = await session.scalar(select(func.count()).select_from(User))
    if user_count and user_count > 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Public registration is disabled. Ask an admin to invite you.",
        )

    existing = await session.execute(select(User).where(User.username == data.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already registered")

    user = User(
        username=data.username,
        email=data.email or f"{data.username}@local",
        hashed_password=hash_password(data.password),
        is_admin=True,
        role=UserRole.admin.value,
    )
    session.add(user)
    await session.flush()

    default_rule = NotificationRule(
        user_id=user.id,
        name="All Events",
        object_types=[],
        named_object_ids=[],
        camera_ids=[],
        channels={"push": True, "email": False},
        debounce_seconds=300,
        enabled=True,
    )
    session.add(default_rule)

    sess = await _create_session(session, user, request)
    register_successful_login(user)
    await audit(
        session,
        action="user.bootstrap_admin",
        actor=user,
        target_type="user",
        target_id=user.id,
        request=request,
    )
    await session.commit()

    return TokenResponse(
        access_token=create_access_token(user.id, user.username, sess.id),
        refresh_token=create_refresh_token(user.id, sess.id),
    )


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    must_change_password: bool = False



# --- Login with TOTP support ---
class LoginStep(BaseModel):
    step: str
    session_id: int | None = None
    user_id: int | None = None
    temp_token: str | None = None

@router.post("/login", response_model=LoginResponse)
async def login(
    data: UserLogin,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(User).where(User.username == data.username))
    user = result.scalar_one_or_none()

    if not user:
        await audit(
            session,
            action="auth.login_failed",
            actor_username=data.username,
            detail={"reason": "unknown_user"},
            request=request,
            commit=True,
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.disabled:
        await audit(session, action="auth.login_blocked", actor=user, detail={"reason": "disabled"}, request=request, commit=True)
        raise HTTPException(status_code=403, detail="Account disabled")

    if is_locked_out(user):
        await audit(session, action="auth.login_blocked", actor=user, detail={"reason": "locked"}, request=request, commit=True)
        raise HTTPException(
            status_code=423,  # Locked
            detail=f"Account locked. Try again at {user.locked_until.isoformat()}",
        )

    if not verify_password(data.password, user.hashed_password):
        register_failed_login(user)
        await audit(session, action="auth.login_failed", actor=user, detail={"reason": "bad_password", "attempts": user.failed_login_attempts}, request=request)
        await session.commit()
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # If TOTP is enabled, require a second step
    if user.totp_enabled:
        # Issue a temp token (JWT with type=totp)
        from core.config import settings
        from jose import jwt
        from jose.exceptions import JWTError
        from datetime import timedelta, datetime, timezone
        expire = datetime.now(timezone.utc) + timedelta(minutes=5)
        payload = {"sub": str(user.id), "exp": expire, "type": "totp"}
        temp_token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        return LoginResponse(
            access_token="",
            refresh_token="",
            must_change_password=user.must_change_password,
            token_type="totp",
            temp_token=temp_token,
        )

    register_successful_login(user)
    sess = await _create_session(session, user, request)
    await audit(session, action="auth.login_success", actor=user, target_type="session", target_id=sess.id, request=request)
    await session.commit()

    return LoginResponse(
        access_token=create_access_token(user.id, user.username, sess.id),
        refresh_token=create_refresh_token(user.id, sess.id),
        must_change_password=user.must_change_password,
    )


# --- TOTP login step ---
class TOTPLoginRequest(BaseModel):
    temp_token: str
    token: str

@router.post("/login/totp", response_model=LoginResponse)
async def login_totp(
    data: TOTPLoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    # Validate temp token
    from core.config import settings
    from jose import jwt
    from jose.exceptions import JWTError
    from datetime import datetime, timezone
    try:
        payload = jwt.decode(data.temp_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "totp":
            raise HTTPException(401, "Invalid token type")
        user_id = int(payload["sub"])
    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.totp_enabled or not user.totp_secret:
        raise HTTPException(401, "TOTP not enabled")
    if not verify_totp(data.token, user.totp_secret):
        raise HTTPException(401, "Invalid TOTP code")

    register_successful_login(user)
    sess = await _create_session(session, user, request)
    await audit(session, action="auth.login_success", actor=user, target_type="session", target_id=sess.id, request=request)
    await session.commit()

    return LoginResponse(
        access_token=create_access_token(user.id, user.username, sess.id),
        refresh_token=create_refresh_token(user.id, sess.id),
        must_change_password=user.must_change_password,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(data: TokenRefresh, session: AsyncSession = Depends(get_session)):
    payload = decode_token(data.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")

    sid = payload.get("sid")
    if sid is not None:
        sess_result = await session.execute(select(UserSession).where(UserSession.id == int(sid)))
        sess = sess_result.scalar_one_or_none()
        if sess is None or sess.revoked_at is not None:
            raise HTTPException(status_code=401, detail="Session revoked")

    user_id = int(payload["sub"])
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.disabled:
        raise HTTPException(status_code=403, detail="Account disabled")

    return TokenResponse(
        access_token=create_access_token(user.id, user.username, sid),
        refresh_token=create_refresh_token(user.id, sid),
    )


@router.post("/logout")
async def logout(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Revoke the session this token belongs to."""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        try:
            payload = decode_token(auth_header.split(" ", 1)[1])
            sid = payload.get("sid")
            if sid is not None:
                sess_result = await session.execute(select(UserSession).where(UserSession.id == int(sid)))
                sess = sess_result.scalar_one_or_none()
                if sess and sess.revoked_at is None:
                    sess.revoked_at = datetime.now(timezone.utc)
                    await audit(session, action="auth.logout", actor=user, target_type="session", target_id=sid, request=request)
        except Exception:
            pass
    await session.commit()
    return {"message": "Logged out"}


@router.get("/me", response_model=UserResponse)
async def get_profile(user: User = Depends(get_current_user)):
    # Ensure totp_enabled is present in response
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_admin=user.is_admin,
        role=user.role,
        theme=user.theme,
        must_change_password=user.must_change_password,
        disabled=user.disabled,
        last_login_at=user.last_login_at,
        created_at=user.created_at,
        totp_enabled=getattr(user, "totp_enabled", False),
    )


@router.get("/me/permissions")
async def get_my_permissions(user: User = Depends(get_current_user)):
    return {
        "role": user_role(user),
        "permissions": permissions_for(user),
        "must_change_password": user.must_change_password,
    }


class ProfileUpdate(BaseModel):
    theme: str | None = None


@router.patch("/me", response_model=UserResponse)
async def update_profile(
    data: ProfileUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if data.theme is not None:
        if data.theme not in ("light", "dark", "system"):
            raise HTTPException(status_code=400, detail="theme must be light, dark or system")
        user.theme = data.theme
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


@router.put("/password")
async def change_password(
    data: PasswordChange,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not verify_password(data.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password")

    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if verify_password(data.new_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="New password must differ from the current one")

    user.hashed_password = hash_password(data.new_password)
    user.must_change_password = False
    session.add(user)
    await audit(session, action="user.password_changed", actor=user, target_type="user", target_id=user.id, request=request)
    await session.commit()
    return {"message": "Password updated"}


# ----- Session management (own sessions) ----------------------------------


@router.get("/sessions")
async def list_my_sessions(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(UserSession).where(UserSession.user_id == user.id).order_by(UserSession.last_seen_at.desc())
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


@router.delete("/sessions/{session_id}")
async def revoke_my_session(
    session_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(UserSession).where(UserSession.id == session_id))
    sess = result.scalar_one_or_none()
    if not sess or sess.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.revoked_at is None:
        sess.revoked_at = datetime.now(timezone.utc)
        await audit(session, action="auth.session_revoked", actor=user, target_type="session", target_id=session_id, request=request)
    await session.commit()
    return {"message": "Session revoked"}
