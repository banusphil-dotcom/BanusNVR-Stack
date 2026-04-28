"""BanusNas — SQLAlchemy async database engine and session."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from core.config import settings

# Tuned pool for DXP2800 (4 cores): keep connections warm, avoid churn
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=5,         # Reduced from 10 — N100 has limited connections
    max_overflow=10,     # Reduced from 20
    pool_timeout=30,
    pool_recycle=1800,   # Recycle connections every 30 min
    pool_pre_ping=True,  # Verify connections aren't stale before checkout
    poolclass=AsyncAdaptedQueuePool,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session():
    async with async_session() as session:
        yield session


async def init_db():
    from models.schemas import Base
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Add columns that may be missing from existing tables
    _migrations = [
        ("cameras", "detection_settings", "JSONB"),
        ("cameras", "ptz_mode", "BOOLEAN DEFAULT false"),
        ("events", "group_key", "VARCHAR(100)"),
        ("users", "theme", "VARCHAR(16) DEFAULT 'system' NOT NULL"),
        # --- User management Phase 1 ---
        ("users", "role", "VARCHAR(20) DEFAULT 'viewer' NOT NULL"),
        ("users", "failed_login_attempts", "INTEGER DEFAULT 0 NOT NULL"),
        ("users", "locked_until", "TIMESTAMPTZ"),
        ("users", "must_change_password", "BOOLEAN DEFAULT false NOT NULL"),
        ("users", "last_login_at", "TIMESTAMPTZ"),
        ("users", "disabled", "BOOLEAN DEFAULT false NOT NULL"),
    ]
    async with engine.begin() as conn:
        for table, column, col_type in _migrations:
            try:
                await conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_type}"
                ))
            except Exception:
                pass  # column already exists
        # Ensure index on events.group_key
        try:
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_events_group_key ON events (group_key)"
            ))
        except Exception:
            pass

        # Backfill role from legacy is_admin flag (only on rows where role is
        # still default — a no-op once anyone has been explicitly assigned).
        try:
            await conn.execute(text(
                "UPDATE users SET role = 'admin' WHERE is_admin = true AND role = 'viewer'"
            ))
        except Exception:
            pass
