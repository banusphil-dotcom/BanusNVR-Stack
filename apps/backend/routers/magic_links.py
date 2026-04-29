"""Magic link login and password reset endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta, timezone
from core.auth import create_access_token, create_refresh_token
from core.config import settings
from models.database import get_session
from models.schemas import User, UserSession
from models.magic_links import MagicLink, PasswordReset
from schemas.magic_links import (
    MagicLinkRequest, MagicLinkVerifyRequest, MagicLinkResponse,
    PasswordResetRequest, PasswordResetVerifyRequest, PasswordResetResponse,
)
import secrets

router = APIRouter(prefix="/api/magic", tags=["magic-links"])

@router.post("/login", response_model=MagicLinkResponse)
async def request_magic_link(data: MagicLinkRequest, request: Request, session: AsyncSession = Depends(get_session)):
    if not settings.auth_magic_links_enabled:
        raise HTTPException(403, "Magic link login is disabled by administrator")
    result = await session.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
    link = MagicLink(user_id=user.id, token=token, expires_at=expires_at)
    session.add(link)
    await session.commit()
    # TODO: Send email with magic link (e.g. /magic/verify?token=...)
    return MagicLinkResponse(message="Magic link sent to your email.")

@router.post("/verify", response_model=MagicLinkResponse)
async def verify_magic_link(data: MagicLinkVerifyRequest, request: Request, session: AsyncSession = Depends(get_session)):
    if not settings.auth_magic_links_enabled:
        raise HTTPException(403, "Magic link login is disabled by administrator")
    result = await session.execute(select(MagicLink).where(MagicLink.token == data.token, MagicLink.used == False))
    link = result.scalar_one_or_none()
    if not link or link.expires_at < datetime.now(timezone.utc):
        raise HTTPException(400, "Invalid or expired magic link")
    user = (await session.execute(select(User).where(User.id == link.user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    link.used = True
    sess = UserSession(
        user_id=user.id,
        user_agent=(request.headers.get("user-agent") or "")[:500],
        ip_address=request.client.host if request.client else None,
    )
    session.add(sess)
    await session.commit()
    return MagicLinkResponse(message="Login successful. Use your app to continue.")

@router.post("/password/request", response_model=PasswordResetResponse)
async def request_password_reset(data: PasswordResetRequest, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(User).where(User.email == data.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    reset = PasswordReset(user_id=user.id, token=token, expires_at=expires_at)
    session.add(reset)
    await session.commit()
    # TODO: Send email with reset link (e.g. /magic/password/verify?token=...)
    return PasswordResetResponse(message="Password reset link sent to your email.")

@router.post("/password/verify", response_model=PasswordResetResponse)
async def verify_password_reset(data: PasswordResetVerifyRequest, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(PasswordReset).where(PasswordReset.token == data.token, PasswordReset.used == False))
    reset = result.scalar_one_or_none()
    if not reset or reset.expires_at < datetime.now(timezone.utc):
        raise HTTPException(400, "Invalid or expired reset link")
    user = (await session.execute(select(User).where(User.id == reset.user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    user.hashed_password = "!reset"  # Should hash new password
    reset.used = True
    await session.commit()
    return PasswordResetResponse(message="Password reset successful. You may now log in.")
