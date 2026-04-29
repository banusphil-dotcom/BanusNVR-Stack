"""BanusNas — Application configuration from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field



class Settings(BaseSettings):
    # Auth method toggles
    auth_totp_enabled: bool = Field(default=True, description="Enable TOTP (2FA)")
    auth_webauthn_enabled: bool = Field(default=True, description="Enable WebAuthn (biometrics/passkeys)")
    auth_oidc_enabled: bool = Field(default=False, description="Enable OIDC (Single Sign-On)")
    auth_api_tokens_enabled: bool = Field(default=True, description="Enable API tokens")
    auth_magic_links_enabled: bool = Field(default=False, description="Enable magic link login")
    # Database
    database_url: str = Field(default="postgresql+asyncpg://banusnas:changeme@db:5432/banusnas")

    # JWT
    jwt_secret_key: str = Field(default="changeme")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30)
    jwt_refresh_token_expire_days: int = Field(default=7)

    # go2rtc (embedded in Frigate — kept for direct access if needed)
    go2rtc_api_url: str = Field(default="http://frigate:1984")

    # Frigate NVR
    frigate_url: str = Field(default="http://frigate:5000")

    # VAPID (Web Push)
    vapid_private_key: str = Field(default="")
    vapid_public_key: str = Field(default="")
    vapid_claim_email: str = Field(default="mailto:admin@banusnas.local")

    # SMTP
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    smtp_from: str = Field(default="banusnas@example.com")
    smtp_tls: bool = Field(default=True)

    # Storage
    recordings_path: str = Field(default="/recordings")
    snapshots_path: str = Field(default="/recordings/snapshots")
    hot_storage_path: str = Field(default="")  # SSD hot storage root (e.g. /livenvr); empty = disabled
    hot_storage_recordings_hours: int = Field(default=48)  # keep recordings on SSD for N hours
    hot_storage_snapshots_hours: int = Field(default=72)  # keep snapshots on SSD for N hours
    retention_events_days: int = Field(default=30)
    retention_continuous_days: int = Field(default=7)
    retention_snapshots_days: int = Field(default=90)

    # Detector settings (kept for recognition service model paths)
    detector_model_path: str = Field(default="/models/yolo26n.onnx")
    detector_device: str = Field(default="GPU")
    detector_input_size: int = Field(default=640)
    detector_confidence_threshold: float = Field(default=0.25)
    detector_nms_iou_threshold: float = Field(default=0.45)
    detector_person_min_confidence: float = Field(default=0.35)
    detector_animal_min_confidence: float = Field(default=0.30)
    coral_enabled: bool = Field(default=False)
    coral_confidence_threshold: float = Field(default=0.40)
    coral_cnn_model_path: str = Field(default="/models/mobilenetv2_features_edgetpu.tflite")
    coral_yolo_model_path: str = Field(default="")

    # AI Summary (local Ollama + optional deep ML)
    ollama_url: str = Field(default="http://ollama:11434")
    ollama_model: str = Field(default="qwen2.5:0.5b")
    ollama_vision_model: str = Field(default="")
    deep_ml_url: str = Field(default="https://ml.banusphotos.com")

    # MQTT (ring-mqtt discovery)
    mqtt_host: str = Field(default="mqtt")
    mqtt_port: int = Field(default=1883)

    # Ring RTSP credentials (from ring-mqtt config)
    ring_rtsp_user: str = Field(default="ring")
    ring_rtsp_password: str = Field(default="ringpass")

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
