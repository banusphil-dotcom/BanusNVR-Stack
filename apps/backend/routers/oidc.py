"""OIDC (Google, GitHub, Generic) login endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
from typing import List
from core.auth import create_access_token, create_refresh_token, get_current_user
from core.config import settings
from models.database import get_session
from models.oidc import OIDCAccount
from models.schemas import User, UserSession
from schemas.oidc import OIDCLoginRequest, OIDCAccountResponse, OIDCAccountListResponse
import httpx, os

router = APIRouter(prefix="/api/oidc", tags=["oidc"])

# Example provider configs (should be loaded from config/env in production)
PROVIDERS = {
    "google": {
        "client_id": os.getenv("GOOGLE_CLIENT_ID", ""),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET", ""),
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "userinfo_url": "https://openidconnect.googleapis.com/v1/userinfo",
    },
    "github": {
        "client_id": os.getenv("GITHUB_CLIENT_ID", ""),
        "client_secret": os.getenv("GITHUB_CLIENT_SECRET", ""),
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "userinfo_url": "https://api.github.com/user",
    },
}

@router.post("/login", response_model=OIDCAccountResponse)
async def oidc_login(
    data: OIDCLoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    if not settings.auth_oidc_enabled:
        raise HTTPException(403, "OIDC is disabled by administrator")
    provider = PROVIDERS.get(data.provider)
    if not provider:
        raise HTTPException(400, "Unknown provider")
    # Exchange code for token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            provider["token_url"],
            data={
                "client_id": provider["client_id"],
                "client_secret": provider["client_secret"],
                "code": data.code,
                "grant_type": "authorization_code",
                "redirect_uri": data.redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise HTTPException(401, "OIDC token exchange failed")
        # Get user info
        userinfo_resp = await client.get(
            provider["userinfo_url"],
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo = userinfo_resp.json()
        email = userinfo.get("email") or userinfo.get("login")
        subject = userinfo.get("sub") or userinfo.get("id")
        if not subject:
            raise HTTPException(401, "OIDC userinfo missing subject")
    # Find or create user
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        user = User(
            username=email,
            email=email,
            hashed_password="!oidc",
            is_admin=False,
            role="viewer",
        )
        session.add(user)
        await session.flush()
    # Link OIDC account
    result = await session.execute(select(OIDCAccount).where(OIDCAccount.provider == data.provider, OIDCAccount.subject == str(subject)))
    oidc = result.scalar_one_or_none()
    if not oidc:
        oidc = OIDCAccount(
            user_id=user.id,
            provider=data.provider,
            subject=str(subject),
            email=email,
            created_at=datetime.now(timezone.utc),
        )
        session.add(oidc)
        await session.flush()
    # Create session
    sess = UserSession(
        user_id=user.id,
        user_agent=(request.headers.get("user-agent") or "")[:500],
        ip_address=request.client.host if request.client else None,
    )
    session.add(sess)
    await session.commit()
    return OIDCAccountResponse(
        id=oidc.id,
        provider=oidc.provider,
        email=oidc.email,
        created_at=oidc.created_at,
    )

@router.get("/accounts", response_model=OIDCAccountListResponse)
async def list_oidc_accounts(
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(OIDCAccount).where(OIDCAccount.user_id == user.id))
    accounts = result.scalars().all()
    return OIDCAccountListResponse(accounts=[
        OIDCAccountResponse(
            id=a.id,
            provider=a.provider,
            email=a.email,
            created_at=a.created_at,
        ) for a in accounts
    ])
