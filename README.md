# BanusNVR

**A self-hosted, AI-powered Network Video Recorder for your home or small business.**

BanusNVR turns a NAS, a mini-PC, or any Linux box into a private, intelligent
CCTV system. It wraps [Frigate](https://frigate.video) with a modern web app,
on-device person / pet / vehicle recognition, AI-generated event narratives,
push notifications, and a hardened multi-user authentication stack including
**passkeys (Touch ID, Windows Hello, FaceID)** and **TOTP two-factor auth**.

Everything runs in your house. Video, faces, recordings, and notifications
never leave your network unless you explicitly enable a Cloudflare Tunnel for
remote access.

---

## What it does

- **Live streams** — low-latency MSE / WebRTC for every camera, on phone or desktop.
- **Recordings & timeline** — 24×7 segmented recording with retention policies, instant scrubbing, and per-event clips.
- **Object detection** — Frigate handles person / car / dog / cat / bird / etc. on Coral, OpenVINO, or CPU.
- **Smart recognition** — InsightFace face recognition + a ReID model identify *who* (or which pet / vehicle) is in the frame.
- **Daily summaries** — a local Ollama LLM summarises the day's events in plain English.
- **Deep Hunt** — natural-language search over historical events ("dog in driveway after 9pm").
- **Push notifications** — Web Push (VAPID) to phones and desktops with rich previews and per-rule muting.
- **Ring integration** — optional `ring-mqtt` bridge brings doorbells and cams into the same UI.
- **Cloudflare Tunnel** — secure remote access, no port-forwarding, no public IP required.
- **PWA** — installable on iOS / Android / desktop with offline shell + background notifications.
- **Multi-user with full RBAC** — admin / operator / viewer roles, audit log, per-camera permissions.

---

## Architecture

```
                   ┌──────────────┐    MQTT     ┌──────────┐
   RTSP cameras ──►│   Frigate    │────────────►│  mqtt    │
                   └──────┬───────┘             └────┬─────┘
                          │ recordings                │
                          ▼                           ▼
                   ┌──────────────┐            ┌────────────┐
                   │  recordings/ │◄───────────│   api      │ (FastAPI)
                   └──────────────┘            │  + ML      │
                                               └─────┬──────┘
                                                     │
                                       ┌─────────────┼─────────────┐
                                       ▼             ▼             ▼
                                  ┌────────┐   ┌─────────┐   ┌──────────┐
                                  │ Postgres│   │ Ollama  │   │  Web UI  │
                                  └────────┘   └─────────┘   └──────────┘
```

| Service          | Image                                           | Purpose                              |
| ---------------- | ----------------------------------------------- | ------------------------------------ |
| `web`            | `ghcr.io/<owner>/banusnvr-web`                  | PWA + nginx reverse proxy            |
| `api`            | `ghcr.io/<owner>/banusnvr-api`                  | FastAPI + recognition pipeline       |
| `frigate`        | `ghcr.io/blakeblackshear/frigate:stable`        | Detection, tracking, recording       |
| `db`             | `postgres:16-alpine`                            | Application database                 |
| `mqtt`           | `eclipse-mosquitto:2`                           | Frigate ↔ API event bus              |
| `ollama`         | `ollama/ollama:latest`                          | Local LLM for narratives             |
| `ring-mqtt`      | `tsightler/ring-mqtt:latest`                    | Ring camera bridge (optional)        |
| `cloudflared`    | `cloudflare/cloudflared:latest`                 | Cloudflare Tunnel (optional)         |
| `setup-wizard`   | `ghcr.io/<owner>/banusnvr-setup-wizard`         | First-run web installer (profile)    |

The optional **ML server** ships separately so it can run on a GPU host:

```bash
docker compose -f docker-compose.ml.yml up -d
# then set DEEP_ML_URL=http://<ml-host>:8765 in your .env
```

See [docs/ML-SERVER.md](docs/ML-SERVER.md).

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/banusphil-dotcom/BanusNVR-Stack.git
cd BanusNVR-Stack

# 2. Run the web setup wizard (writes .env for you)
docker compose --profile setup up -d
# → open http://<your-host-ip>:8090, fill the form, click Save.

# 3. Stop the wizard
docker compose --profile setup down

# 4. Pull & launch the full stack
docker compose pull
docker compose up -d

# 5. Open the app
#    http://<your-host-ip>:8080
```

The first user that registers becomes the admin automatically.

Prefer to skip the wizard? Copy `.env.example` to `.env`, edit it, then
`docker compose up -d`.

---

## Authentication

BanusNVR ships with a layered authentication system. Admins enable / disable
each method from **Settings → Account & Security** (or via
`PUT /api/auth/settings`).

| Method                  | Default | Description                                                       |
| ----------------------- | ------- | ----------------------------------------------------------------- |
| Username + password     | always  | Bcrypt-hashed, account lockout after repeated failures.           |
| **TOTP** (2FA)          | on      | Any RFC 6238 authenticator (Google Authenticator, 1Password, …).  |
| **Passkeys / WebAuthn** | on      | Touch ID, Face ID, Windows Hello, YubiKey, Android biometrics.    |
| API tokens              | on      | Long-lived bearer tokens for scripts and Home Assistant.          |
| OIDC (SSO)              | off     | Sign in with an external identity provider (stub — see docs).     |
| Magic links             | off     | E-mail one-time login links (requires SMTP — stub).               |

### Setting up TOTP (2FA)

1. Sign in.
2. Open **Settings → Account & Security → Two-factor authentication**.
3. Click **Enable 2FA**, scan the QR code with your authenticator app, enter the
   6-digit code, and confirm.
4. From now on, every sign-in will ask for your code after the password step.

### Setting up Passkeys (biometric / hardware)

Passkeys let you sign in with Touch ID, Face ID, Windows Hello, an Android
fingerprint sensor, or a hardware key like YubiKey — no password required.

**Requirements:**

- A modern browser (Chrome / Edge / Safari / Firefox).
- The site must be served over **HTTPS** (or `localhost` for development).
- Set `WEBAUTHN_RP_ID` in your `.env` to the **public hostname** users connect
  to (e.g. `nvr.example.com`). It must match exactly — no scheme, no port.
  When using a Cloudflare Tunnel, set it to your tunnel hostname.
- `WEBAUTHN_RP_NAME` is the friendly name shown in the OS prompt
  (default: `BanusNVR`).

**Enrolling a passkey:**

1. Sign in normally (password ± TOTP).
2. **Settings → Account & Security → Passkeys / Biometrics**.
3. Type a name for the device ("iPhone", "Work laptop", "YubiKey 5") and
   click **Add passkey**.
4. Approve the OS prompt.

**Signing in with a passkey:**

On the login screen, click **Sign in with passkey / biometrics**. You can
either type your username first (server will offer that user's keys) or leave
it blank to use a discoverable credential — the OS will let you pick.

### Recovery

If you lose your authenticator app or all your passkeys, an admin (or anyone
with shell access to the database) can clear them:

```bash
docker exec -it banusnvr-db psql -U banusnvr -d banusnvr

# disable TOTP for one user
UPDATE users SET totp_enabled = false, totp_secret = NULL WHERE username = 'alice';

# remove all passkeys for one user
DELETE FROM webauthn_credentials WHERE user_id = (SELECT id FROM users WHERE username = 'alice');

# unlock a locked account
UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE username = 'alice';
```

---

## Configuration

Every option is documented in `.env.example` and surfaced by the setup
wizard. Key categories:

- **Storage** — host paths for recordings, snapshots, models, hot storage.
- **Database / JWT / VAPID / SMTP** — secrets (auto-generated by the wizard).
- **Cloudflare Tunnel** — paste a token from Zero Trust to enable.
- **Ring** — RTSP credentials for the optional Ring bridge.
- **Hardware** — Intel iGPU vs Coral USB vs CPU detector.
- **WebAuthn** — `WEBAUTHN_RP_ID`, `WEBAUTHN_RP_NAME` (see Passkeys above).
- **Auth toggles** — `AUTH_TOTP_ENABLED`, `AUTH_WEBAUTHN_ENABLED`,
  `AUTH_OIDC_ENABLED`, `AUTH_API_TOKENS_ENABLED`, `AUTH_MAGIC_LINKS_ENABLED`.

For deeper docs see:

- [docs/INSTALL.md](docs/INSTALL.md)
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- [docs/CLOUDFLARE.md](docs/CLOUDFLARE.md)
- [docs/ML-SERVER.md](docs/ML-SERVER.md)
- [docs/UPGRADE.md](docs/UPGRADE.md)

---

## Updating

```bash
git pull
docker compose pull
docker compose up -d
```

Pin to a specific release by setting `BANUSNVR_TAG=v1.2.3` in `.env`.

---

## Building locally (developers)

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

Source for each image lives under [`apps/`](apps).

Repository layout:

```
apps/
  backend/         FastAPI + recognition pipeline (Python 3.12)
  frontend/        Vite + React + TS + Tailwind PWA
  ml-server/       Optional GPU offload server
  setup-wizard/    First-run installer
config/            Sample configs for Frigate, mosquitto, nginx, cloudflared
docs/              Long-form docs
docker-compose.yml Production stack
```

---

## License

MIT — see [LICENSE](LICENSE).
