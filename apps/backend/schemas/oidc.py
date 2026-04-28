from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class OIDCProviderConfig(BaseModel):
    provider: str
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    userinfo_url: str
    scopes: list[str] = ["openid", "email", "profile"]

class OIDCLoginRequest(BaseModel):
    provider: str
    code: str
    redirect_uri: str

class OIDCAccountResponse(BaseModel):
    id: int
    provider: str
    email: Optional[str]
    created_at: datetime

class OIDCAccountListResponse(BaseModel):
    accounts: list[OIDCAccountResponse]
