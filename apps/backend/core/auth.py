
from typing import Optional
import hashlib

from models.api_tokens import APIToken
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status

async def get_user_from_api_token(raw_token: str, session: AsyncSession) -> Optional["User"]:
    """Authenticate using an API token (for Authorization: Bearer ...)."""
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    result = await session.execute(
        select(APIToken).where(APIToken.token_hash == token_hash, APIToken.revoked == False)
    )
    token = result.scalar_one_or_none()
    if not token or (token.expires_at and token.expires_at < datetime.now(timezone.utc)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired API token")
    # Update last_used_at
    token.last_used_at = datetime.now(timezone.utc)
    await session.commit()
    # Return user
    from models.schemas import User
    result = await session.execute(select(User).where(User.id == token.user_id))
    user = result.scalar_one_or_none()
    if not user or getattr(user, "disabled", False):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled")
    return user
import pyotp
# ----- TOTP 2FA helpers ---------------------------------------------------
def generate_totp_secret() -> str:
    return pyotp.random_base32()

def get_totp_uri(username: str, secret: str, issuer: str = "BanusNVR") -> str:
    return pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer)

def verify_totp(token: str, secret: str) -> bool:
    totp = pyotp.TOTP(secret)
    return totp.verify(token, valid_window=1)
"""BanusNas — JWT authentication, lockout, and session validation.

Tokens carry a session id (`sid`) so revoking a row in `user_sessions`
immediately invalidates every JWT issued under that session — without
having to rotate the global signing key. This is what powers the "kick
device off" admin action.
"""


from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Query, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from models.database import get_session

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Account lockout policy
MAX_FAILED_LOGINS = 5
LOCKOUT_MINUTES = 15


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: int, username: str, session_id: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload: dict = {"sub": str(user_id), "username": username, "exp": expire, "type": "access"}
    if session_id is not None:
        payload["sid"] = session_id
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: int, session_id: int | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_token_expire_days)
    payload: dict = {"sub": str(user_id), "exp": expire, "type": "refresh"}
    if session_id is not None:
        payload["sid"] = session_id
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def _validate_session(payload: dict, session: AsyncSession) -> None:
    """If the JWT carries a session id, ensure it hasn't been revoked.

    Tokens issued before sessions existed (legacy) have no `sid` and we
    accept them — they'll be rotated naturally on next refresh.
    """
    sid = payload.get("sid")
    if sid is None:
        return
    from models.schemas import UserSession

    result = await session.execute(select(UserSession).where(UserSession.id == int(sid)))
    sess = result.scalar_one_or_none()
    if sess is None or sess.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session revoked")
    sess.last_seen_at = datetime.now(timezone.utc)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_session),
):
    payload = decode_token(credentials.credentials)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    await _validate_session(payload, session)

    user_id = int(payload["sub"])

    from models.schemas import User

    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if getattr(user, "disabled", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")
    return user


async def get_user_from_token_param(
    token: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """Auth via query parameter — used for image src tags that can't send headers."""
    if not token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token required")
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token type")
    await _validate_session(payload, session)
    user_id = int(payload["sub"])
    from models.schemas import User
    result = await session.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User not found")
    if getattr(user, "disabled", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")
    return user


async def get_current_user_ws(websocket: WebSocket, token: Optional[str] = Query(None)):
    """Authenticate a WebSocket connection via query-string token.

    Note: WS handshake intentionally does NOT validate session revocation
    to keep the connect cheap. If a session is revoked, the next API call
    on the same token will fail and the client will reconnect.
    """
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        if payload.get("type") != "access":
            return None
        return int(payload["sub"])
    except JWTError:
        return None


# ----- Lockout helpers (used by login flow) -------------------------------


def is_locked_out(user) -> bool:
    if user.locked_until is None:
        return False
    locked_until = user.locked_until
    if locked_until.tzinfo is None:
        locked_until = locked_until.replace(tzinfo=timezone.utc)
    return locked_until > datetime.now(timezone.utc)


def register_failed_login(user) -> None:
    """Increment failed counter and lock the account if threshold hit."""
    user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
    if user.failed_login_attempts >= MAX_FAILED_LOGINS:
        user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
        user.failed_login_attempts = 0


def register_successful_login(user) -> None:
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = datetime.now(timezone.utc)
