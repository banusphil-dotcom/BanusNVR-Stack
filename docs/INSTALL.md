# Installation

## Requirements

- Linux host with Docker Engine 24+ and the Compose v2 plugin.
- ~20 GB free disk for images + model cache.
- Recommended: Intel CPU with iGPU (uses VAAPI + OpenVINO).
  Other detectors (`cpu`, `tensorrt`, `edgetpu`) are supported via
  `FRIGATE_DETECTOR` in `.env`.
- Recordings storage on a fast disk (NVMe ideal). Network shares work but slow.

## Step-by-step

### 1. Clone & prepare host paths

```bash
git clone https://github.com/banusphil-dotcom/BanusNVR-Stack.git
cd BanusNVR-Stack
mkdir -p data/recordings data/snapshots data/models data/livenvr
```

### 2. Provide configuration

**Option A — web wizard (recommended):**

```bash
docker compose --profile setup up -d
# open http://<host-ip>:8090
docker compose --profile setup down
```

**Option B — manual:**

```bash
cp .env.example .env
$EDITOR .env
```

Generate secrets:

```bash
# JWT secret
openssl rand -hex 32

# VAPID keypair (push notifications)
docker run --rm node:20-alpine npx web-push generate-vapid-keys
```

### 3. Frigate camera config

Copy the example and edit it (or use the BanusNVR cameras UI which will
overwrite this file when you save changes):

```bash
cp config/frigate/config.example.yml config/frigate/config.yml
$EDITOR config/frigate/config.yml
```

### 4. Pull & start

```bash
docker compose pull
docker compose up -d
docker compose logs -f api
```

Open `http://<host-ip>:8080` in a browser.

### 5. First login

The first user to register becomes the admin. If you want to lock down
registration after that, set `ALLOW_REGISTRATION=false` in `.env` and
`docker compose up -d`.

## Hardware acceleration

| Hardware             | `FRIGATE_DETECTOR` | Notes                                          |
| -------------------- | ------------------ | ---------------------------------------------- |
| Intel iGPU           | `openvino`         | Default. Needs `/dev/dri` (already mounted).   |
| NVIDIA GPU           | `tensorrt`         | Switch image tag and add nvidia runtime.       |
| Coral USB            | `edgetpu`          | Set `CORAL_ENABLED=true`, plug into USB.       |
| CPU only             | `cpu`              | Slow — fine for 1-2 cameras.                   |

## Backup

The data that matters lives in:

- `./data/` — recordings, snapshots, models.
- `./config/frigate/config.yml` — Frigate camera config.
- Docker named volumes `db_data`, `mqtt_data`, `ollama_data`, `ring_data`.

Use `scripts/backup.sh` (TBD) or `docker run --rm -v db_data:/v -v $(pwd):/b alpine tar czf /b/db_backup.tgz /v`.
