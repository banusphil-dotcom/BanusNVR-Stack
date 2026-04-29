"""VAPID key management: generate, load, and persist VAPID keys in DB."""
import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from py_vapid import Vapid01 as Vapid
from models.schemas import SystemSettings

VAPID_KEY_DB_KEY = "vapid_keys"

async def get_vapid_keys(session: AsyncSession) -> Optional[dict]:
    result = await session.execute(
        select(SystemSettings).where(SystemSettings.key == VAPID_KEY_DB_KEY)
    )
    row = result.scalar_one_or_none()
    if row and row.value:
        return row.value
    return None

async def set_vapid_keys(session: AsyncSession, public_key: str, private_key: str, claim_email: str):
    # Upsert
    result = await session.execute(
        select(SystemSettings).where(SystemSettings.key == VAPID_KEY_DB_KEY)
    )
    row = result.scalar_one_or_none()
    value = {"public_key": public_key, "private_key": private_key, "claim_email": claim_email}
    if row:
        row.value = value
        session.add(row)
    else:
        row = SystemSettings(key=VAPID_KEY_DB_KEY, value=value)
        session.add(row)
    await session.commit()

async def ensure_vapid_keys(session: AsyncSession, claim_email: str) -> dict:
    """Ensure VAPID keys exist in DB, generate if missing. Returns dict with public_key, private_key, claim_email."""
    keys = await get_vapid_keys(session)
    if keys and keys.get("public_key") and keys.get("private_key"):
        return keys
    # Generate new keys
    logging.info("Generating new VAPID key pair...")
    vapid = Vapid()
    vapid.generate_keys()
    public_key = vapid.public_key.decode()
    private_key = vapid.private_key.decode()
    await set_vapid_keys(session, public_key, private_key, claim_email)
    return {"public_key": public_key, "private_key": private_key, "claim_email": claim_email}