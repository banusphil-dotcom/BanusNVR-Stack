# ML Server (optional GPU offload)

The BanusNVR API ships with **on-device** models that run on the same host
as the API. For a small number of cameras this is fine. If you want to:

- Run heavier YOLO variants (YOLOv8m / YOLOv9-c)
- Use GPU-accelerated InsightFace
- Centralize inference for multiple BanusNVR instances

…then deploy the companion ML server on a host with a GPU and point your
BanusNVR at it.

## HTTP contract

The API expects the following endpoints (all JSON):

| Method | Path                       | Body                                                                |
| ------ | -------------------------- | ------------------------------------------------------------------- |
| GET    | `/v1/health`               | —                                                                   |
| POST   | `/v1/detect`               | `{ image (b64 jpeg), confidence_threshold, target_classes? }`       |
| POST   | `/v1/embedding`            | `{ image, model: "cnn" \| "reid" }`                                 |
| POST   | `/v1/faces`                | `{ image, min_face_size }`                                          |
| POST   | `/v1/chat/completions`     | OpenAI-compatible (proxied to `LLM_BACKEND_URL`)                    |

A reference CPU implementation lives in [`apps/ml-server/`](../apps/ml-server).
The published image `ghcr.io/<owner>/banusnvr-ml:latest` is built from it.

## Deploy

On the GPU host:

```bash
git clone https://github.com/banusphil-dotcom/BanusNVR-Stack.git
cd BanusNVR-Stack

# Optional auth (recommended if exposed off-LAN)
echo "ML_API_KEY=$(openssl rand -hex 24)" > .env.ml
# echo "LLM_BACKEND_URL=http://ollama:11434" >> .env.ml   # if proxying chat

docker compose --env-file .env.ml -f docker-compose.ml.yml pull
docker compose --env-file .env.ml -f docker-compose.ml.yml up -d
```

Then on the BanusNVR host, set in `.env`:

```env
DEEP_ML_URL=http://<ml-host-ip>:8765
# If you set ML_API_KEY above, also set:
DEEP_ML_API_KEY=<the same key>
```

Restart BanusNVR: `docker compose up -d`.

## NVIDIA GPU

Install [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
on the ML host, then uncomment the `deploy.resources.reservations.devices`
block in `docker-compose.ml.yml`.

## Implementing your own

Anything that satisfies the contract above will work. A minimal Python
implementation is ~100 lines (see `apps/ml-server/server.py`). You can also
implement it in Triton, TorchServe, or vLLM behind an adapter.
