"""API Token management endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import List
from core.auth import get_current_user, hash_password
from core.config import settings
from models.database import get_session
from models.api_tokens import APIToken
from models.schemas import User
from schemas.api_tokens import APITokenCreate, APITokenResponse, APITokenListResponse
import secrets, hashlib

router = APIRouter(prefix="/api/tokens", tags=["api-tokens"])

@router.post("/", response_model=APITokenResponse, status_code=201)
async def create_token(
    data: APITokenCreate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_api_tokens_enabled:
        raise HTTPException(403, "API tokens are disabled by administrator")
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    scopes = ",".join(data.scopes or [])
    token = APIToken(
        user_id=user.id,
        name=data.name,
        token_hash=token_hash,
        created_at=datetime.now(timezone.utc),
        expires_at=data.expires_at,
        scopes=scopes,
    )
    session.add(token)
    await session.commit()
    return APITokenResponse(
        id=token.id,
        name=token.name,
        token=raw_token,
        created_at=token.created_at,
        expires_at=token.expires_at,
        last_used_at=None,
        scopes=data.scopes or [],
        revoked=False,
    )

@router.get("/", response_model=APITokenListResponse)
async def list_tokens(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_api_tokens_enabled:
        raise HTTPException(403, "API tokens are disabled by administrator")
    result = await session.execute(select(APIToken).where(APIToken.user_id == user.id))
    tokens = result.scalars().all()
    return APITokenListResponse(tokens=[
        APITokenResponse(
            id=t.id,
            name=t.name,
            created_at=t.created_at,
            expires_at=t.expires_at,
            last_used_at=t.last_used_at,
            scopes=t.scopes.split(",") if t.scopes else [],
            revoked=t.revoked,
        ) for t in tokens
    ])

@router.delete("/{token_id}", status_code=204)
async def revoke_token(
    token_id: int,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_api_tokens_enabled:
        raise HTTPException(403, "API tokens are disabled by administrator")
    result = await session.execute(select(APIToken).where(APIToken.id == token_id, APIToken.user_id == user.id))
    token = result.scalar_one_or_none()
    if not token:
        raise HTTPException(404, "Token not found")
    token.revoked = True
    await session.commit()
    return
