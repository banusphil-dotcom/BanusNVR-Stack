"""
BanusNVR — First-run web setup wizard.

Renders a single-page form, validates input, and writes a complete .env file
to the host project directory (mounted at /workspace).
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ENV_FILE = Path(os.environ.get("ENV_FILE_PATH", "/workspace/.env"))
ENV_EXAMPLE = Path(os.environ.get("ENV_EXAMPLE_PATH", "/workspace/.env.example"))

app = FastAPI(title="BanusNVR Setup")
templates = Jinja2Templates(directory="/app/templates")
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


# ---------------------------------------------------------------------------
# Field schema — drives both the form and the .env writer
# ---------------------------------------------------------------------------
SECTIONS: list[dict] = [
    {
        "title": "Storage paths",
        "help": "Where recordings, snapshots, and models will live on this host.",
        "fields": [
            ("DATA_ROOT", "Data root", "./data", "text"),
            ("RECORDINGS_PATH", "Recordings path", "./data/recordings", "text"),
            ("SNAPSHOTS_PATH", "Snapshots path", "./data/snapshots", "text"),
            ("MODELS_PATH", "Models path", "./data/models", "text"),
            ("HOT_STORAGE_PATH", "Hot (NVMe) storage path", "./data/livenvr", "text"),
        ],
    },
    {
        "title": "Web access",
        "help": "Ports exposed on the host.",
        "fields": [
            ("WEB_HTTP_PORT", "Web UI port", "8080", "number"),
            ("FRIGATE_PORT", "Frigate UI port (debug)", "5000", "number"),
            ("GO2RTC_RTSP_PORT", "go2rtc RTSP port", "8554", "number"),
            ("GO2RTC_WEBRTC_PORT", "go2rtc WebRTC port", "8555", "number"),
            ("GO2RTC_API_PORT", "go2rtc API port", "1984", "number"),
        ],
    },
    {
        "title": "Database",
        "help": "PostgreSQL credentials. The DATABASE_URL is generated automatically.",
        "fields": [
            ("POSTGRES_USER", "DB user", "banusnvr", "text"),
            ("POSTGRES_PASSWORD", "DB password", "", "password"),
            ("POSTGRES_DB", "DB name", "banusnvr", "text"),
        ],
    },
    {
        "title": "Push notifications (optional)",
        "help": (
            "Generate VAPID keys with: "
            "<code>docker run --rm node:20-alpine npx web-push generate-vapid-keys</code>"
        ),
        "fields": [
            ("VAPID_PUBLIC_KEY", "VAPID public key", "", "text"),
            ("VAPID_PRIVATE_KEY", "VAPID private key", "", "password"),
            ("VAPID_CLAIM_EMAIL", "Contact email", "mailto:admin@example.com", "text"),
        ],
    },
    {
        "title": "Email notifications (optional)",
        "help": "Leave blank to disable email alerts.",
        "fields": [
            ("SMTP_HOST", "SMTP host", "", "text"),
            ("SMTP_PORT", "SMTP port", "587", "number"),
            ("SMTP_USER", "SMTP user", "", "text"),
            ("SMTP_PASSWORD", "SMTP password", "", "password"),
            ("SMTP_FROM", "From address", "", "text"),
        ],
    },
    {
        "title": "Retention",
        "help": "Days to keep events / continuous recordings.",
        "fields": [
            ("RETENTION_EVENTS_DAYS", "Event days", "30", "number"),
            ("RETENTION_CONTINUOUS_DAYS", "Continuous days", "0", "number"),
            ("RETENTION_SNAPSHOTS_DAYS", "Snapshot days", "30", "number"),
        ],
    },
    {
        "title": "Cloudflare Tunnel (optional)",
        "help": (
            "For secure public access without port-forwarding. "
            "Get a token from the Cloudflare Zero Trust dashboard. Leave blank to skip."
        ),
        "fields": [
            ("CLOUDFLARE_TUNNEL_TOKEN", "Tunnel token", "", "password"),
        ],
    },
    {
        "title": "Ring cameras (optional)",
        "help": "Only relevant if you have Ring cameras.",
        "fields": [
            ("RING_RTSP_USER", "Ring RTSP user", "ring", "text"),
            ("RING_RTSP_PASSWORD", "Ring RTSP password", "", "password"),
        ],
    },
    {
        "title": "Remote ML server (optional)",
        "help": (
            "URL of an external GPU-backed ML server. See <code>docs/ML-SERVER.md</code>. "
            "Leave blank to use only the embedded models."
        ),
        "fields": [
            ("DEEP_ML_URL", "ML server URL", "", "text"),
        ],
    },
    {
        "title": "Hardware acceleration",
        "help": "Tune for your host. Defaults work for most Intel iGPU systems.",
        "fields": [
            ("FRIGATE_DETECTOR", "Frigate detector", "openvino", "text"),
            ("CORAL_ENABLED", "Coral USB Edge TPU connected? (true/false)", "false", "text"),
            ("TZ", "Timezone", "Europe/London", "text"),
        ],
    },
]


def _gen_secret(n: int = 32) -> str:
    return secrets.token_hex(n)


def _is_configured() -> bool:
    return ENV_FILE.exists() and ENV_FILE.stat().st_size > 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if _is_configured():
        return templates.TemplateResponse(
            "done.html",
            {"request": request, "env_path": str(ENV_FILE)},
        )
    return templates.TemplateResponse(
        "wizard.html",
        {"request": request, "sections": SECTIONS},
    )


@app.post("/save")
async def save(request: Request):
    form = await request.form()
    values: dict[str, str] = {k: str(v).strip() for k, v in form.items()}

    # Auto-generate secrets
    jwt_secret = _gen_secret(32)
    pg_user = values.get("POSTGRES_USER", "banusnvr") or "banusnvr"
    pg_pass = values.get("POSTGRES_PASSWORD") or _gen_secret(16)
    pg_db = values.get("POSTGRES_DB", "banusnvr") or "banusnvr"
    values["POSTGRES_PASSWORD"] = pg_pass
    values["JWT_SECRET_KEY"] = jwt_secret
    values["DATABASE_URL"] = f"postgresql+asyncpg://{pg_user}:{pg_pass}@db:5432/{pg_db}"
    values.setdefault("BANUSNVR_REGISTRY", "ghcr.io/banusphil-dotcom")
    values.setdefault("BANUSNVR_TAG", "latest")
    values.setdefault("SETUP_WIZARD_PORT", "8090")
    values.setdefault("OLLAMA_URL", "http://ollama:11434")
    values.setdefault("OLLAMA_MODEL", "qwen2.5:0.5b")
    values.setdefault("JWT_ALGORITHM", "HS256")
    values.setdefault("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
    values.setdefault("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30")
    values.setdefault("SMTP_TLS", "true")

    lines = [
        "# Generated by BanusNVR setup wizard.",
        "# To re-run: stop containers, delete .env, "
        "and start the 'setup' profile again.",
        "",
    ]
    for section in SECTIONS:
        lines.append(f"# --- {section['title']} ---")
        for key, _, _, _ in section["fields"]:
            v = values.get(key, "")
            lines.append(f"{key}={v}")
        lines.append("")
    # Emit auto-derived keys
    lines.append("# --- Auto-generated secrets ---")
    for k in ("BANUSNVR_REGISTRY", "BANUSNVR_TAG", "SETUP_WIZARD_PORT",
              "POSTGRES_PASSWORD", "JWT_SECRET_KEY", "JWT_ALGORITHM",
              "JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "JWT_REFRESH_TOKEN_EXPIRE_DAYS",
              "DATABASE_URL", "OLLAMA_URL", "OLLAMA_MODEL", "SMTP_TLS"):
        lines.append(f"{k}={values.get(k, '')}")

    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RedirectResponse("/done", status_code=303)


@app.get("/done", response_class=HTMLResponse)
async def done(request: Request):
    return templates.TemplateResponse(
        "done.html",
        {"request": request, "env_path": str(ENV_FILE)},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "configured": _is_configured()}
