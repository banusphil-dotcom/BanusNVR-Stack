from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, EmailStr

class MagicLinkRequest(BaseModel):
    email: EmailStr

class MagicLinkVerifyRequest(BaseModel):
    token: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetVerifyRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)

class MagicLinkResponse(BaseModel):
    message: str

class PasswordResetResponse(BaseModel):
    message: str
