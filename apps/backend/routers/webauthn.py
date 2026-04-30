"""WebAuthn (passkey / biometric) endpoints — fido2 v2.x.

Flow
====
Registration (authenticated user adding a new device):
  1. POST /api/webauthn/register/begin     → server issues challenge + options
  2. browser navigator.credentials.create(options)
  3. POST /api/webauthn/register/complete  → server verifies, stores public key

Login (anyone with a registered credential):
  1. POST /api/webauthn/login/begin        → server issues challenge + allow-list
  2. browser navigator.credentials.get(options)
  3. POST /api/webauthn/login/complete     → server verifies, returns JWTs

Challenges are kept in process memory; expire after 5 min.
"""

from __future__ import annotations

import base64
import secrets
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Request
from fido2.server import Fido2Server
from fido2.webauthn import (
    AttestedCredentialData,
    PublicKeyCredentialRpEntity,
    PublicKeyCredentialUserEntity,
    UserVerificationRequirement,
)
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.audit import audit
from core.auth import create_access_token, create_refresh_token, get_current_user
from core.config import settings
from models.database import get_session
from models.schemas import User, UserSession
from models.webauthn import WebAuthnCredential
from schemas.webauthn import (
    WebAuthnCredentialListResponse,
    WebAuthnCredentialResponse,
)

router = APIRouter(prefix="/api/webauthn", tags=["webauthn"])


# ───────────────────────── helpers ─────────────────────────

_CHALLENGE_TTL_S = 300
# challenge_id → (state, expires_at, mode, user_id_or_none)
_pending: dict[str, tuple[Any, float, str, int | None]] = {}


def _gc_pending() -> None:
    now = time.time()
    for k in [k for k, (_, exp, _, _) in _pending.items() if exp < now]:
        _pending.pop(k, None)


def _resolve_rp(request: Request) -> tuple[str, str]:
    origin = request.headers.get("origin") or ""
    origin_host = urlparse(origin).hostname or ""
    rp_id = settings.webauthn_rp_id.strip() or origin_host
    if not rp_id:
        raise HTTPException(400, "Cannot determine WebAuthn relying-party id (no Origin header and webauthn_rp_id not configured)")
    return rp_id, settings.webauthn_rp_name or "BanusNVR"


def _server(rp_id: str, rp_name: str) -> Fido2Server:
    return Fido2Server(PublicKeyCredentialRpEntity(name=rp_name, id=rp_id))


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def _user_handle(user: User) -> bytes:
    return user.id.to_bytes(8, "big")


async def _user_credentials(session: AsyncSession, user_id: int) -> list[WebAuthnCredential]:
    res = await session.execute(
        select(WebAuthnCredential).where(WebAuthnCredential.user_id == user_id)
    )
    return list(res.scalars().all())


def _to_attested(cred: WebAuthnCredential) -> AttestedCredentialData:
    """Stored `public_key` column holds the raw AttestedCredentialData bytes."""
    return AttestedCredentialData(cred.public_key)


def _new_challenge_id() -> str:
    return secrets.token_urlsafe(24)


def _jsonify(obj: Any) -> Any:
    """Recursively turn fido2 dicts/dataclasses with bytes into JSON-safe form."""
    if isinstance(obj, (bytes, bytearray)):
        return _b64url(bytes(obj))
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    val = getattr(obj, "value", None)
    if val is not None and isinstance(val, (str, int, bytes)):
        return _jsonify(val)
    if hasattr(obj, "items"):  # fido2 Options behave like dicts
        try:
            return {k: _jsonify(v) for k, v in obj.items()}
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        d = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return _jsonify(d)
    return obj


# ───────────────────────── registration ─────────────────────────


class RegisterBeginResponse(BaseModel):
    challenge_id: str
    publicKey: dict


@router.post("/register/begin", response_model=RegisterBeginResponse)
async def register_begin(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_webauthn_enabled:
        raise HTTPException(403, "WebAuthn is disabled by administrator")

    rp_id, rp_name = _resolve_rp(request)
    server = _server(rp_id, rp_name)

    existing = await _user_credentials(session, user.id)
    exclude = [_to_attested(c) for c in existing]

    user_entity = PublicKeyCredentialUserEntity(
        id=_user_handle(user),
        name=user.username,
        display_name=user.username,
    )

    options, state = server.register_begin(
        user_entity,
        credentials=exclude,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    cid = _new_challenge_id()
    _pending[cid] = (state, time.time() + _CHALLENGE_TTL_S, "register", user.id)
    _gc_pending()

    return RegisterBeginResponse(challenge_id=cid, publicKey=_jsonify(options.public_key))


class RegisterCompleteRequest(BaseModel):
    challenge_id: str
    name: str
    id: str
    rawId: str
    type: str = "public-key"
    response: dict  # {clientDataJSON, attestationObject, transports?}


@router.post("/register/complete", response_model=WebAuthnCredentialResponse, status_code=201)
async def register_complete(
    data: RegisterCompleteRequest,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_webauthn_enabled:
        raise HTTPException(403, "WebAuthn is disabled by administrator")

    pending = _pending.pop(data.challenge_id, None)
    if not pending or pending[1] < time.time() or pending[2] != "register" or pending[3] != user.id:
        raise HTTPException(400, "Challenge expired or invalid")
    state = pending[0]

    rp_id, rp_name = _resolve_rp(request)
    server = _server(rp_id, rp_name)

    # Build the dict expected by fido2.server.register_complete with raw bytes.
    response_dict = {
        "id": data.id,
        "rawId": _b64url_decode(data.rawId),
        "type": data.type,
        "response": {
            "clientDataJSON": _b64url_decode(data.response["clientDataJSON"]),
            "attestationObject": _b64url_decode(data.response["attestationObject"]),
        },
    }

    try:
        auth_data = server.register_complete(state, response_dict)
    except Exception as e:
        raise HTTPException(400, f"Attestation verification failed: {e}")

    if auth_data.credential_data is None:
        raise HTTPException(400, "No credential data in attestation")

    transports = data.response.get("transports") or []
    cred = WebAuthnCredential(
        user_id=user.id,
        name=(data.name or "Passkey")[:100],
        credential_id=auth_data.credential_data.credential_id,
        public_key=bytes(auth_data.credential_data),
        sign_count=auth_data.counter,
        transports=",".join(transports)[:100],
        is_backup=bool(getattr(auth_data, "flags", 0) & 0x10),  # FLAG.BACKUP_STATE
        created_at=datetime.now(timezone.utc),
    )
    session.add(cred)
    await audit(session, action="webauthn.register", actor=user, target_type="webauthn", target_id=cred.id, request=request)
    await session.commit()

    return WebAuthnCredentialResponse(
        id=cred.id,
        name=cred.name,
        created_at=cred.created_at,
        last_used_at=None,
        transports=transports,
        is_backup=cred.is_backup,
    )


# ───────────────────────── authentication ─────────────────────────


class LoginBeginRequest(BaseModel):
    username: str | None = None  # omit for usernameless / discoverable creds


class LoginBeginResponse(BaseModel):
    challenge_id: str
    publicKey: dict


@router.post("/login/begin", response_model=LoginBeginResponse)
async def login_begin(
    data: LoginBeginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_webauthn_enabled:
        raise HTTPException(403, "WebAuthn is disabled by administrator")

    rp_id, rp_name = _resolve_rp(request)
    server = _server(rp_id, rp_name)

    creds: list[AttestedCredentialData] = []
    user_id_hint: int | None = None
    if data.username:
        res = await session.execute(select(User).where(User.username == data.username))
        u = res.scalar_one_or_none()
        if u:
            user_id_hint = u.id
            creds = [_to_attested(c) for c in await _user_credentials(session, u.id)]

    options, state = server.authenticate_begin(
        credentials=creds,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    cid = _new_challenge_id()
    _pending[cid] = (state, time.time() + _CHALLENGE_TTL_S, "login", user_id_hint)
    _gc_pending()

    return LoginBeginResponse(challenge_id=cid, publicKey=_jsonify(options.public_key))


class LoginCompleteRequest(BaseModel):
    challenge_id: str
    id: str
    rawId: str
    type: str = "public-key"
    response: dict  # {clientDataJSON, authenticatorData, signature, userHandle?}


class LoginCompleteResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    must_change_password: bool = False


@router.post("/login/complete", response_model=LoginCompleteResponse)
async def login_complete(
    data: LoginCompleteRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_webauthn_enabled:
        raise HTTPException(403, "WebAuthn is disabled by administrator")

    pending = _pending.pop(data.challenge_id, None)
    if not pending or pending[1] < time.time() or pending[2] != "login":
        raise HTTPException(400, "Challenge expired or invalid")
    state = pending[0]

    rp_id, rp_name = _resolve_rp(request)
    server = _server(rp_id, rp_name)

    raw_id = _b64url_decode(data.rawId)

    res = await session.execute(
        select(WebAuthnCredential).where(WebAuthnCredential.credential_id == raw_id)
    )
    cred = res.scalar_one_or_none()
    if not cred:
        raise HTTPException(401, "Unknown credential")

    user_res = await session.execute(select(User).where(User.id == cred.user_id))
    user = user_res.scalar_one_or_none()
    if not user or user.disabled:
        raise HTTPException(401, "User not available")

    response_dict = {
        "id": data.id,
        "rawId": raw_id,
        "type": data.type,
        "response": {
            "clientDataJSON": _b64url_decode(data.response["clientDataJSON"]),
            "authenticatorData": _b64url_decode(data.response["authenticatorData"]),
            "signature": _b64url_decode(data.response["signature"]),
            "userHandle": _b64url_decode(data.response["userHandle"]) if data.response.get("userHandle") else b"",
        },
    }

    try:
        result_cred = server.authenticate_complete(
            state,
            credentials=[_to_attested(cred)],
            response=response_dict,
        )
    except Exception as e:
        raise HTTPException(401, f"Signature verification failed: {e}")

    cred.last_used_at = datetime.now(timezone.utc)
    session.add(cred)

    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = datetime.now(timezone.utc)
    session.add(user)

    sess = UserSession(
        user_id=user.id,
        user_agent=(request.headers.get("user-agent") or "")[:500],
        ip_address=(request.client.host if request.client else None),
    )
    session.add(sess)
    await session.flush()

    await audit(
        session,
        action="auth.login_success",
        actor=user,
        target_type="session",
        target_id=sess.id,
        request=request,
        detail={"method": "webauthn"},
    )
    await session.commit()
    _ = result_cred  # silence unused

    return LoginCompleteResponse(
        access_token=create_access_token(user.id, user.username, sess.id),
        refresh_token=create_refresh_token(user.id, sess.id),
        must_change_password=user.must_change_password,
    )


# ───────────────────────── manage credentials ─────────────────────────


@router.get("/", response_model=WebAuthnCredentialListResponse)
async def list_credentials(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    creds = await _user_credentials(session, user.id)
    return WebAuthnCredentialListResponse(credentials=[
        WebAuthnCredentialResponse(
            id=c.id,
            name=c.name,
            created_at=c.created_at,
            last_used_at=c.last_used_at,
            transports=c.transports.split(",") if c.transports else [],
            is_backup=c.is_backup,
        )
        for c in creds
    ])


@router.delete("/{credential_id}", status_code=204)
async def delete_credential(
    credential_id: int,
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    res = await session.execute(
        select(WebAuthnCredential).where(
            WebAuthnCredential.id == credential_id,
            WebAuthnCredential.user_id == user.id,
        )
    )
    cred = res.scalar_one_or_none()
    if not cred:
        raise HTTPException(404, "Credential not found")
    await session.delete(cred)
    await audit(session, action="webauthn.delete", actor=user, target_type="webauthn", target_id=credential_id, request=request)
    await session.commit()
    return
