"""BanusNas — SQLAlchemy ORM models."""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# --- Enums ---


class CameraType(str, enum.Enum):
    tapo = "tapo"
    ring = "ring"
    hikvision = "hikvision"
    onvif = "onvif"
    rtsp = "rtsp"
    other = "other"


class RecordingMode(str, enum.Enum):
    continuous = "continuous"
    motion = "motion"
    events = "events"
    disabled = "disabled"


class ObjectCategory(str, enum.Enum):
    person = "person"
    pet = "pet"
    vehicle = "vehicle"
    other = "other"


class EventType(str, enum.Enum):
    motion = "motion"
    object_detected = "object_detected"
    object_recognized = "object_recognized"


# --- Models ---


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    push_subscription: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    theme: Mapped[str] = mapped_column(String(16), default="system", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    notification_rules: Mapped[list["NotificationRule"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    camera_type: Mapped[CameraType] = mapped_column(Enum(CameraType), nullable=False)
    connection_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    recording_mode: Mapped[RecordingMode] = mapped_column(Enum(RecordingMode), default=RecordingMode.motion)
    detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    detection_objects: Mapped[list] = mapped_column(JSONB, default=lambda: ["person", "cat", "dog", "car"])
    detection_confidence: Mapped[float] = mapped_column(Float, default=0.5)
    detection_settings: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    ptz_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    ptz_config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    zones: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    snapshot_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    events: Mapped[list["Event"]] = relationship(back_populates="camera", cascade="all, delete-orphan")


class CameraCredential(Base):
    """Reusable camera login (username/password) that can be applied to many cameras.

    Stored in plaintext (same trust level as `Camera.connection_config`, which is
    already plaintext credentials per camera). Encryption-at-rest is the
    operator's responsibility (volume / DB level).
    """
    __tablename__ = "camera_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    camera_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # tapo / hikvision / onvif / ... or None = any
    username: Mapped[str] = mapped_column(String(200), nullable=False)
    password: Mapped[str] = mapped_column(String(500), nullable=False, default="")
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class NamedObject(Base):
    __tablename__ = "named_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    category: Mapped[ObjectCategory] = mapped_column(Enum(ObjectCategory), nullable=False)
    reference_image_count: Mapped[int] = mapped_column(Integer, default=0)
    compreface_subject_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    body_embedding: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSONB, nullable=True)  # Learned soft biometrics
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    events: Mapped[list["Event"]] = relationship(back_populates="named_object")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), nullable=False)
    object_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    named_object_id: Mapped[int | None] = mapped_column(ForeignKey("named_objects.id", ondelete="SET NULL"), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    bbox: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recording_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    group_key: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_extra: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    camera: Mapped["Camera"] = relationship(back_populates="events")
    named_object: Mapped["NamedObject | None"] = relationship(back_populates="events")


class NotificationRule(Base):
    __tablename__ = "notification_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    object_types: Mapped[list] = mapped_column(JSONB, default=list)
    named_object_ids: Mapped[list] = mapped_column(JSONB, default=list)
    camera_ids: Mapped[list] = mapped_column(JSONB, default=list)
    schedule: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    channels: Mapped[dict] = mapped_column(JSONB, default=lambda: {"push": True, "email": False})
    debounce_seconds: Mapped[int] = mapped_column(Integer, default=300)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship(back_populates="notification_rules")


class SentNotification(Base):
    __tablename__ = "sent_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channel: Mapped[str] = mapped_column(String(20), nullable=False)  # "push" or "email"
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[str] = mapped_column(String(1000), nullable=False)
    camera_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    object_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class SystemSettings(Base):
    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(200), unique=True, nullable=False, index=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)


class DailySummary(Base):
    __tablename__ = "daily_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # YYYY-MM-DD
    summary_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "morning" or "evening"
    data: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
