"""BanusNas — FastAPI application entry point.

v2: Frigate NVR handles detection/tracking/recording.
    This backend handles recognition, identity, notifications, and frontend API.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from models.database import init_db, get_session, async_session
from models.schemas import NotificationRule, User as UserModel
from sqlalchemy import select
from services.recognition_service import recognition_service
from services.face_service import face_service
from services.frigate_bridge import frigate_bridge, event_bus
from services.storage_manager import storage_manager
from services.notification_engine import notification_engine
from services.daily_summary import daily_summary_service

from routers import auth, cameras, events, recordings, training, search, notifications, system, credentials, users, audit_logs
from routers import summary as summary_router
from routers import ring as ring_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("banusnas")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle.

    v2: Frigate handles detection/tracking/recording.
    We only start recognition services + Frigate MQTT bridge.
    """
    logger.info("BanusNas NVR v2 (Frigate) starting up...")

    # Database
    await init_db()
    logger.info("Database initialized")

    # Schema migration: add body_embedding column if missing
    async with async_session() as session:
        try:
            from sqlalchemy import text
            await session.execute(text(
                "ALTER TABLE named_objects ADD COLUMN IF NOT EXISTS body_embedding JSONB"
            ))
            await session.commit()
            logger.info("Schema migration: body_embedding column ensured")
        except Exception as e:
            logger.warning("Schema migration skipped: %s", e)
            await session.rollback()

    # Ensure all users have at least one notification rule
    async with async_session() as session:
        users = (await session.execute(select(UserModel))).scalars().all()
        for u in users:
            existing = await session.execute(
                select(NotificationRule).where(NotificationRule.user_id == u.id).limit(1)
            )
            if not existing.scalar_one_or_none():
                session.add(NotificationRule(
                    user_id=u.id,
                    name="All Events",
                    object_types=[],
                    named_object_ids=[],
                    camera_ids=[],
                    channels={"push": True, "email": False},
                    debounce_seconds=300,
                    enabled=True,
                ))
                logger.info("Created default notification rule for user %s", u.username)
        await session.commit()

    # Face detection + recognition (InsightFace: SCRFD + ArcFace)
    await face_service.start()
    logger.info("Face service: %s", "ready" if face_service.is_available else "disabled")

    # Recognition service (MobileNetV2 CNN + Person ReID)
    await recognition_service.start()
    logger.info("Recognition: CNN %s, Person ReID %s",
                "ready" if recognition_service._cnn_ready else "disabled",
                "ready" if recognition_service.reid_available else "disabled")

    # Load saved performance settings (ML offload, etc.)
    try:
        from routers.system import _load_performance_settings, _apply_performance_settings
        async with async_session() as session:
            perf = await _load_performance_settings(session)
            _apply_performance_settings(perf)
            logger.info("Performance settings loaded: ML offload %s → %s",
                        "enabled" if perf.ml_offload_enabled else "disabled",
                        perf.ml_offload_url if perf.ml_offload_enabled else "n/a")
    except Exception as e:
        logger.warning("Failed to load performance settings: %s", e)

    # Coral Edge TPU (CNN feature extraction — offloads MobileNetV2 from CPU)
    if settings.coral_enabled:
        from services.coral_backend import coral_backend
        coral_ok = await asyncio.to_thread(
            coral_backend.start,
            settings.coral_yolo_model_path,
            settings.coral_cnn_model_path,
        )
        logger.info("Coral Edge TPU: %s", "ready" if coral_ok else "disabled (no hardware or models)")
    else:
        logger.info("Coral Edge TPU: disabled (CORAL_ENABLED=false)")

    # Wire notification engine to Frigate bridge
    frigate_bridge.set_notification_engine(notification_engine)

    # Storage cleanup loop (snapshots only — Frigate manages recordings)
    await storage_manager.start()
    logger.info("Storage manager started")

    # Daily summary scheduler
    await daily_summary_service.start()
    logger.info("Daily summary scheduler started")

    # Periodic profile integrity check (runs once daily at 03:00)
    async def _profile_integrity_loop():
        """Background loop: run profile integrity checks at 03:00 daily."""
        while True:
            try:
                now = datetime.now()
                # Calculate seconds until next 03:00
                target = now.replace(hour=3, minute=0, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)
                wait_secs = (target - now).total_seconds()
                await asyncio.sleep(wait_secs)

                logger.info("Starting periodic profile integrity checks...")
                from routers.training import run_all_profile_checks
                results = await run_all_profile_checks()
                flagged = [r for r in results if not r.get("consistent", True)]
                logger.info("Profile integrity: checked %d profiles, %d flagged",
                            len(results), len(flagged))
                for f in flagged:
                    logger.warning("Profile integrity issue: %s — outliers=%s conf=%d",
                                   f["name"], f.get("outlier_indices", []), f.get("confidence", 0))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.warning("Profile integrity loop error", exc_info=True)
                await asyncio.sleep(3600)  # retry in 1h on error

    _integrity_task = asyncio.create_task(_profile_integrity_loop())
    logger.info("Profile integrity checker scheduled (03:00 daily)")

    # Start Frigate MQTT event bridge (connects to MQTT, subscribes to frigate/events)
    await frigate_bridge.start()
    logger.info("Frigate bridge started — detection/recognition pipeline active")

    # Register all DB cameras in go2rtc (dynamic streams survive API restarts)
    try:
        from services.stream_manager import stream_manager
        from routers.cameras import build_source_url
        async with async_session() as session:
            from models.schemas import Camera
            result = await session.execute(select(Camera).where(Camera.enabled == True))
            cams = result.scalars().all()
            registered = 0
            for cam in cams:
                source_url = build_source_url(cam.camera_type.value, cam.connection_config or {})
                if source_url and source_url not in ("", "rtsp://admin:@:554/stream1"):
                    stream_name = f"camera_{cam.id}"
                    try:
                        await stream_manager.add_stream(stream_name, source_url)
                        registered += 1
                    except Exception as e:
                        logger.warning("Failed to register stream for camera %s: %s", cam.id, e)
            logger.info("Registered %d/%d camera streams in go2rtc", registered, len(cams))
    except Exception as e:
        logger.warning("Failed to register camera streams at startup: %s", e)

    # Regenerate Frigate config from current cameras so any detector/model
    # changes shipped with the API image are applied without needing the user
    # to add/edit a camera. Safe no-op if config dir isn't mounted.
    try:
        from services.frigate_config import deploy_frigate_config
        from models.schemas import Camera as _Cam
        async with async_session() as session:
            result = await session.execute(select(_Cam).order_by(_Cam.id))
            cams = result.scalars().all()
            resp = await deploy_frigate_config(cams)
            logger.info("Frigate config regenerated at startup: %s", resp.get("message"))
    except Exception as e:
        logger.warning("Failed to regenerate Frigate config at startup: %s", e)

    # Start Ring MQTT background listener for device discovery
    try:
        from routers.ring import _start_ring_listener
        _start_ring_listener()
        logger.info("Ring MQTT listener started")
    except Exception as e:
        logger.warning("Ring MQTT listener failed to start: %s", e)

    logger.info("BanusNas NVR v2 ready — Frigate handles detection/tracking/recording")

    yield

    # Shutdown
    logger.info("BanusNas NVR shutting down...")
    _integrity_task.cancel()
    await frigate_bridge.stop()
    await daily_summary_service.stop()
    await storage_manager.stop()
    logger.info("Shutdown complete")


app = FastAPI(
    title="BanusNas NVR",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(recordings.router)
app.include_router(training.router)
app.include_router(search.router)
app.include_router(search.public_router)
app.include_router(notifications.router)
app.include_router(system.router)
app.include_router(summary_router.router)
app.include_router(ring_router.router)
app.include_router(credentials.router)
app.include_router(users.router)
app.include_router(audit_logs.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "banusnas-nvr"}


@app.websocket("/ws/events")
async def websocket_events(ws: WebSocket):
    """Real-time event stream via WebSocket with keepalive ping."""
    await ws.accept()

    queue = event_bus.subscribe()

    async def _sender():
        while True:
            data = await queue.get()
            await ws.send_json(data)

    async def _pinger():
        """Send periodic pings to keep connection alive through NAT/firewalls."""
        while True:
            await asyncio.sleep(25)
            await ws.send_json({"type": "ping"})

    try:
        await asyncio.gather(_sender(), _pinger())
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        event_bus.unsubscribe(queue)
