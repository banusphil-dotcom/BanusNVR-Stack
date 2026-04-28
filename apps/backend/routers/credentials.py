"""BanusNas — Saved camera login credentials.

Lets the user save reusable username/password pairs (e.g. "Tapo home account")
that can then be applied when adding cameras manually or used to bulk-test
LAN-scan results.
"""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import get_current_user
from models.database import get_session
from models.schemas import CameraCredential
from schemas.api_schemas import CredentialCreate, CredentialResponse, CredentialUpdate

router = APIRouter(
    prefix="/api/camera-credentials",
    tags=["camera-credentials"],
    dependencies=[Depends(get_current_user)],
)


def _to_response(c: CameraCredential) -> CredentialResponse:
    return CredentialResponse(
        id=c.id,
        name=c.name,
        username=c.username,
        camera_type=c.camera_type,
        notes=c.notes,
        has_password=bool(c.password),
        created_at=c.created_at,
    )


@router.get("", response_model=list[CredentialResponse])
async def list_credentials(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(CameraCredential).order_by(CameraCredential.name))
    return [_to_response(c) for c in result.scalars().all()]


@router.post("", response_model=CredentialResponse, status_code=status.HTTP_201_CREATED)
async def create_credential(data: CredentialCreate, session: AsyncSession = Depends(get_session)):
    # Enforce unique name
    existing = await session.execute(select(CameraCredential).where(CameraCredential.name == data.name))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Credential named '{data.name}' already exists")

    cred = CameraCredential(
        name=data.name,
        username=data.username,
        password=data.password or "",
        camera_type=data.camera_type,
        notes=data.notes,
    )
    session.add(cred)
    await session.commit()
    await session.refresh(cred)
    return _to_response(cred)


@router.put("/{credential_id}", response_model=CredentialResponse)
async def update_credential(
    credential_id: int,
    data: CredentialUpdate,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(CameraCredential).where(CameraCredential.id == credential_id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(status_code=404, detail="Credential not found")

    update_data = data.model_dump(exclude_unset=True)
    # Empty-string password = "leave unchanged" (so editing without re-typing the
    # password works in the UI). Set explicit None to clear.
    if "password" in update_data and update_data["password"] == "":
        update_data.pop("password")

    for key, value in update_data.items():
        setattr(cred, key, value)
    session.add(cred)
    await session.commit()
    await session.refresh(cred)
    return _to_response(cred)


@router.delete("/{credential_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_credential(credential_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(CameraCredential).where(CameraCredential.id == credential_id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    await session.delete(cred)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


class _RevealRequest(BaseModel):
    credential_id: int


@router.post("/reveal-password")
async def reveal_password(data: _RevealRequest, session: AsyncSession = Depends(get_session)):
    """Return the stored password for an authenticated user.

    Used by the camera-add UI to apply a saved login to the live test-connection
    form (which sends username+password directly to go2rtc). Only authenticated
    users can call this — the dependency is enforced router-level.
    """
    result = await session.execute(select(CameraCredential).where(CameraCredential.id == data.credential_id))
    cred = result.scalar_one_or_none()
    if not cred:
        raise HTTPException(status_code=404, detail="Credential not found")
    return {"password": cred.password or ""}
