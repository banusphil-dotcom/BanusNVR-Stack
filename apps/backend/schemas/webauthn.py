from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class WebAuthnCredentialCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    credential_id: str  # base64url
    public_key: str     # base64url
    sign_count: int = 0
    transports: Optional[list[str]] = None
    is_backup: bool = False

class WebAuthnCredentialResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    last_used_at: Optional[datetime] = None
    transports: list[str] = []
    is_backup: bool = False

class WebAuthnCredentialListResponse(BaseModel):
    credentials: list[WebAuthnCredentialResponse]
