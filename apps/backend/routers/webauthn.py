"""WebAuthn (Passkey) registration and authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import List
from core.auth import get_current_user
from models.database import get_session
from models.webauthn import WebAuthnCredential
from models.schemas import User
from schemas.webauthn import WebAuthnCredentialCreate, WebAuthnCredentialResponse, WebAuthnCredentialListResponse
import base64

router = APIRouter(prefix="/api/webauthn", tags=["webauthn"])

@router.post("/register", response_model=WebAuthnCredentialResponse, status_code=201)
async def register_credential(
    data: WebAuthnCredentialCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    cred = WebAuthnCredential(
        user_id=user.id,
        name=data.name,
        credential_id=base64.urlsafe_b64decode(data.credential_id + '=='),
        public_key=base64.urlsafe_b64decode(data.public_key + '=='),
        sign_count=data.sign_count,
        transports=",".join(data.transports or []),
        is_backup=data.is_backup,
        created_at=datetime.now(timezone.utc),
    )
    session.add(cred)
    await session.commit()
    return WebAuthnCredentialResponse(
        id=cred.id,
        name=cred.name,
        created_at=cred.created_at,
        last_used_at=None,
        transports=data.transports or [],
        is_backup=data.is_backup,
    )

@router.get("/", response_model=WebAuthnCredentialListResponse)
async def list_credentials(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(WebAuthnCredential).where(WebAuthnCredential.user_id == user.id))
    creds = result.scalars().all()
    return WebAuthnCredentialListResponse(credentials=[
        WebAuthnCredentialResponse(
            id=c.id,
            name=c.name,
            created_at=c.created_at,
            last_used_at=c.last_used_at,
            transports=c.transports.split(",") if c.transports else [],
            is_backup=c.is_backup,
        ) for c in creds
    ])

@router.delete("/{credential_id}", status_code=204)
async def delete_credential(
    credential_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(WebAuthnCredential).where(WebAuthnCredential.id == credential_id, WebAuthnCredential.user_id == user.id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(404, "Credential not found")
    await session.delete(cred)
    await session.commit()
    return
