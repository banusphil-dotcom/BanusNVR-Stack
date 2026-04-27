# =============================================================================
# BanusNas API — Dockerfile  (v2: Frigate backend)
# =============================================================================

# --- Stage 1: Export MobileNetV2 features to ONNX (for pet/vehicle recognition) ---
FROM python:3.11-slim AS model-export
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 && rm -rf /var/lib/apt/lists/*
# Newer torch.onnx.export eagerly imports onnxscript on call, even for the legacy
# opset path. Install it explicitly so the export step succeeds.
RUN pip install --no-cache-dir torch torchvision onnx onnxscript

# Export MobileNetV2 features-only (1280-dim, no classifier head)
RUN python -c "\
import torch, torchvision; \
m = torchvision.models.mobilenet_v2(weights='DEFAULT'); \
m.classifier = torch.nn.Identity(); \
m.eval(); \
x = torch.randn(1, 3, 224, 224); \
torch.onnx.export(m, x, '/models/mobilenetv2_features.onnx', opset_version=12, \
    input_names=['input'], output_names=['features'])" \
    && mkdir -p /models

# --- Stage 1b: (Coral Edge TPU pre-compilation has been removed) -------------
# The previous stage tried to pre-compile a MobileNetV2 INT8 TFLite for the
# Edge TPU at image build time. It pulled in TensorFlow (~600 MB), depended on
# an externally-hosted libedgetpu release tarball that is no longer reachable,
# and used invalid one-line Python that no longer parses on modern interpreters.
#
# Coral support is OFF by default (CORAL_ENABLED=false). For users who want it,
# the model can be built/downloaded at runtime or supplied via the /models
# bind-mount. See docs/CONFIGURATION.md.

# --- Stage 2: Runtime image ---
FROM python:3.11-slim

# System dependencies: FFmpeg, OpenCV deps, g++ for insightface Cython build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    gnupg \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Intel OpenCL runtime (iGPU access for OpenVINO GPU inference — recognition models)
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" \
    > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Coral Edge TPU runtime install was removed (upstream URL returned 404).
# Install libedgetpu1-std at runtime on the host instead, or rebuild this image
# with a working URL when re-enabling Coral support.

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ai-edge-litert

# Copy models from build stage (MobileNetV2 for pet/vehicle CNN recognition)
RUN mkdir -p /models-builtin /models
COPY --from=model-export /models/mobilenetv2_features.onnx /models-builtin/mobilenetv2_features.onnx

# (Coral Edge TPU model copy removed — see Stage 1b note above.)

# Download InsightFace model pack (buffalo_s: SCRFD + ArcFace MobileFaceNet)
# Root = /models-builtin/insightface → models end up at /models-builtin/insightface/models/buffalo_s/
# Only load detection + recognition modules (skip 3D landmarks 137 MB, genderage, etc.)
RUN python -c "\
import warnings; warnings.filterwarnings('ignore'); \
from insightface.app import FaceAnalysis; \
import os; os.makedirs('/models-builtin/insightface/models', exist_ok=True); \
app = FaceAnalysis(name='buffalo_s', root='/models-builtin/insightface', \
    providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition']); \
app.prepare(ctx_id=-1, det_size=(640, 640)); \
print('InsightFace buffalo_s ready (det + rec only)')" || \
    echo "InsightFace model download deferred to first start"

# Download OpenVINO person ReID model (person-reidentification-retail-0287, FP16)
RUN mkdir -p /models-builtin/person_reid && \
    curl -fsSL -o /models-builtin/person_reid/person-reidentification-retail-0287.xml \
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.xml" && \
    curl -fsSL -o /models-builtin/person_reid/person-reidentification-retail-0287.bin \
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/person-reidentification-retail-0287/FP16/person-reidentification-retail-0287.bin" || \
    echo "Person ReID model download deferred to first start"

# Copy source code (build v14-frigate — change this comment to bust Docker cache)
COPY . .

EXPOSE 8000

# Entrypoint: copy built-in models to /models/ volume if not already present, then start
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--loop", "uvloop"]
