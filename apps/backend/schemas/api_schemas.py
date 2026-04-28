"""BanusNas — Pydantic request/response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# --- Auth ---


class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    email: Optional[str] = None
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenRefresh(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool
    theme: str = "system"
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Cameras ---


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    camera_type: str  # tapo, ring, hikvision, onvif, rtsp, other
    connection_config: dict = Field(default_factory=dict)
    credential_id: Optional[int] = None  # Apply saved username/password
    recording_mode: str = "motion"
    detection_enabled: bool = True
    detection_objects: list[str] = Field(default_factory=lambda: ["person", "cat", "dog", "car"])
    detection_confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    detection_settings: Optional[dict] = None
    ptz_mode: bool = False
    ptz_config: Optional[dict] = None
    zones: Optional[dict] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    camera_type: Optional[str] = None
    connection_config: Optional[dict] = None
    enabled: Optional[bool] = None
    recording_mode: Optional[str] = None
    detection_enabled: Optional[bool] = None
    detection_objects: Optional[list[str]] = None
    detection_confidence: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    detection_settings: Optional[dict] = None
    ptz_mode: Optional[bool] = None
    ptz_config: Optional[dict] = None
    zones: Optional[dict] = None


class CameraResponse(BaseModel):
    id: int
    name: str
    camera_type: str
    connection_config: dict
    enabled: bool
    recording_mode: str
    detection_enabled: bool
    detection_objects: list[str]
    detection_confidence: float
    detection_settings: Optional[dict] = None
    ptz_mode: bool
    ptz_config: Optional[dict] = None
    zones: Optional[dict]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CameraStatusResponse(CameraResponse):
    is_recording: bool = False
    is_detecting: bool = False
    last_snapshot_url: Optional[str] = None


# --- Camera Credentials (saved logins reusable across cameras) ---


class CredentialCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    username: str = Field(min_length=1, max_length=200)
    password: str = Field(default="", max_length=500)
    camera_type: Optional[str] = None  # If set, suggested for matching camera types only
    notes: Optional[str] = Field(default=None, max_length=500)


class CredentialUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    username: Optional[str] = Field(default=None, min_length=1, max_length=200)
    password: Optional[str] = Field(default=None, max_length=500)
    camera_type: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=500)


class CredentialResponse(BaseModel):
    id: int
    name: str
    username: str
    camera_type: Optional[str] = None
    notes: Optional[str] = None
    has_password: bool = True
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Bulk camera-add (used by LAN scan auto-probe results) ---


class BulkCameraAdd(BaseModel):
    cameras: list[CameraCreate]


class AutoProbeDevice(BaseModel):
    ip: str
    camera_type: Optional[str] = None  # If unknown, all credential types are tried
    port: Optional[int] = 554


class AutoProbeRequest(BaseModel):
    devices: list[AutoProbeDevice]
    credential_ids: list[int] = Field(default_factory=list)
    extra_credentials: list[CredentialCreate] = Field(default_factory=list)  # ad-hoc creds (not saved)


class AutoProbeResult(BaseModel):
    ip: str
    camera_type: str
    success: bool
    credential_id: Optional[int] = None
    credential_name: Optional[str] = None
    username: Optional[str] = None
    stream_path: Optional[str] = None
    sub_stream_path: Optional[str] = None
    source_url: Optional[str] = None
    snapshot: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    suggested_name: Optional[str] = None
    error: Optional[str] = None


# --- Events ---


class EventAnnotation(BaseModel):
    name: str
    class_name: str
    bbox: dict
    confidence: Optional[float] = None
    primary: bool = False


class EventResponse(BaseModel):
    id: int
    camera_id: int
    camera_name: Optional[str] = None
    event_type: str
    object_type: Optional[str]
    named_object_id: Optional[int]
    named_object_name: Optional[str] = None
    confidence: Optional[float]
    snapshot_path: Optional[str]
    thumbnail_path: Optional[str]
    bbox: Optional[dict]
    recording_path: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    duration: Optional[float] = None
    gif_url: Optional[str] = None
    annotations: list[EventAnnotation] = Field(default_factory=list)
    narrative: Optional[str] = None

    model_config = {"from_attributes": True}


class EventLabel(BaseModel):
    named_object_id: int


class EventsPage(BaseModel):
    items: list[EventResponse]
    total: int
    page: int
    page_size: int


class EventGroupResponse(BaseModel):
    group_key: str
    camera_id: int
    camera_name: str
    camera_names: list[str] = Field(default_factory=list)
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration: Optional[float] = None
    narrative: str
    names: list[str]
    object_count: int
    primary_event_id: int
    events: list[EventResponse]


class EventGroupsPage(BaseModel):
    groups: list[EventGroupResponse]
    total_groups: int
    page: int
    page_size: int


# --- Named Objects ---


class NamedObjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    category: str  # person, pet, vehicle, other


class NamedObjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    gender: Optional[str] = Field(default=None, pattern=r"^(male|female)$")
    age_group: Optional[str] = Field(default=None, pattern=r"^(child|young_adult|adult|middle_aged|senior)$")
    breed: Optional[str] = Field(default=None, max_length=100)
    color: Optional[str] = Field(default=None, max_length=100)
    markings: Optional[str] = Field(default=None, max_length=100)
    vehicle_type: Optional[str] = Field(default=None, max_length=100)
    make: Optional[str] = Field(default=None, max_length=100)


class NamedObjectResponse(BaseModel):
    id: int
    name: str
    category: str
    reference_image_count: int
    attributes: Optional[dict] = None
    created_at: datetime
    last_seen: Optional[datetime] = None
    last_camera: Optional[str] = None

    model_config = {"from_attributes": True}


class TrainFromEventsRequest(BaseModel):
    event_ids: list[int] = Field(min_length=1)


class CreateAndTrainRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    category: str  # person, pet
    event_ids: list[int] = Field(min_length=1)


# --- Search ---


class LastSeenResponse(BaseModel):
    named_object: NamedObjectResponse
    camera_name: str
    camera_id: int
    timestamp: datetime
    snapshot_url: Optional[str]


class TimelineEntry(BaseModel):
    event_id: int
    camera_id: int
    camera_name: str
    timestamp: datetime
    confidence: Optional[float]
    snapshot_url: Optional[str]
    narrative: Optional[str] = None


# --- Notifications ---


class PushSubscription(BaseModel):
    endpoint: str
    keys: dict


class NotificationRuleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    object_types: list[str] = Field(default_factory=list)
    named_object_ids: list[int] = Field(default_factory=list)
    camera_ids: list[int] = Field(default_factory=list)
    schedule: Optional[dict] = None
    channels: dict = Field(default_factory=lambda: {"push": True, "email": False})
    debounce_seconds: int = Field(default=300, ge=0)


class NotificationRuleUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    object_types: Optional[list[str]] = None
    named_object_ids: Optional[list[int]] = None
    camera_ids: Optional[list[int]] = None
    schedule: Optional[dict] = None
    channels: Optional[dict] = None
    debounce_seconds: Optional[int] = Field(default=None, ge=0)
    enabled: Optional[bool] = None


class NotificationRuleResponse(BaseModel):
    id: int
    name: str
    object_types: list[str]
    named_object_ids: list[int]
    camera_ids: list[int]
    schedule: Optional[dict]
    channels: dict
    debounce_seconds: int
    enabled: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class SentNotificationResponse(BaseModel):
    id: int
    event_id: Optional[int]
    channel: str
    title: str
    body: str
    camera_name: Optional[str]
    object_type: Optional[str]
    read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class NotificationHistoryPage(BaseModel):
    items: list[SentNotificationResponse]
    total: int
    page: int
    page_size: int


# --- System ---


class HardwareWarning(BaseModel):
    level: str  # "info", "warning", "critical"
    category: str  # "cpu", "ram", "gpu", "storage", "cameras"
    message: str
    value: float  # current usage percentage
    limit: Optional[float] = None  # capacity limit if applicable


class HardwareResources(BaseModel):
    cpu_name: str
    cpu_cores: int
    cpu_percent: float
    ram_total_gb: float
    ram_used_gb: float
    ram_percent: float
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_percent: Optional[float] = None
    gpu_inference_device: Optional[str] = None
    coral_available: bool = False
    coral_status: Optional[dict] = None
    detector_type: Optional[str] = None  # "edgetpu" | "openvino" | "cpu" | ...
    detector_devices: list[str] = []      # human-readable list, e.g. ["Coral USB"]
    storage_used_gb: float
    storage_total_gb: float
    storage_percent: float
    cameras_active: int
    cameras_relay: int  # passthrough (no transcode)
    cameras_transcode: int  # needing transcode
    estimated_max_cameras_relay: int
    estimated_max_cameras_transcode: int
    uptime_seconds: float = 0
    warnings: list[HardwareWarning] = []


class SystemStatus(BaseModel):
    cameras_online: int
    cameras_total: int
    active_recordings: int
    events_today: int
    storage_used_gb: float
    storage_total_gb: float
    uptime_seconds: float


class GlobalSettings(BaseModel):
    retention_events_days: int = 30
    retention_continuous_days: int = 7
    retention_snapshots_days: int = 90
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_from: Optional[str] = None


class PerformanceSettings(BaseModel):
    """Tunable performance parameters — adjustable from the Settings UI."""
    # Motion detection
    motion_frame_skip: int = Field(default=10, ge=4, le=30, description="Process every Nth frame for motion (higher = less CPU)")
    motion_blur_kernel: int = Field(default=7, ge=3, le=21, description="Gaussian blur kernel size (odd only)")
    motion_cooldown: float = Field(default=5.0, ge=1.0, le=30.0, description="Seconds between motion triggers")
    # Object tracking (YOLO)
    track_interval: float = Field(default=3.0, ge=1.0, le=10.0, description="Seconds between YOLO inferences per camera")
    enhanced_scan_interval: float = Field(default=30.0, ge=10.0, le=120.0, description="Seconds between enhanced small-object scans")
    yolo_concurrency: int = Field(default=1, ge=1, le=4, description="Max simultaneous YOLO inferences")
    # Recognition pipeline
    max_detection_pipelines: int = Field(default=2, ge=1, le=6, description="Max concurrent recognition/event pipelines")
    # Snapshot quality
    jpeg_quality: int = Field(default=80, ge=40, le=100, description="JPEG quality for saved snapshots")
    # ML offload (remote GPU server)
    ml_offload_enabled: bool = Field(default=False, description="Offload ML inference to remote GPU server")
    ml_offload_url: str = Field(default="https://ml.banusphotos.com", description="Remote ML server URL")
    # Coral Edge TPU
    coral_enabled: bool = Field(default=False, description="Use Coral USB Accelerator for detection/CNN")
    # Preset name (informational)
    preset: Optional[str] = None


class TrainingSettings(BaseModel):
    """Training & recognition reinforcement settings."""
    auto_enroll_enabled: bool = Field(default=True, description="Auto-enroll high-confidence detections as training data")
    auto_enroll_threshold: float = Field(default=0.90, ge=0.50, le=1.0, description="Minimum confidence to auto-enroll (0.50–1.0)")
    training_retention_days: int = Field(default=90, ge=7, le=365, description="Days to keep pinned training detections (7–365)")
    auto_reinforce_cap: int = Field(default=50, ge=10, le=200, description="Max reference images per profile")


class DashboardLayout(BaseModel):
    layout: str = "auto"
    camera_order: list[int] = []
    hidden_cameras: list[int] = []
