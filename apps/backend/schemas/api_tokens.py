from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class APITokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    expires_at: Optional[datetime] = None
    scopes: Optional[list[str]] = None

class APITokenResponse(BaseModel):
    id: int
    name: str
    token: Optional[str] = None  # Only returned on creation
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    scopes: list[str] = []
    revoked: bool = False

class APITokenListResponse(BaseModel):
    tokens: list[APITokenResponse]
