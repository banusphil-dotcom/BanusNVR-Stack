# =============================================================================
# BanusNas API — Dockerfile  (v2: Frigate backend)
# =============================================================================

# --- Stage 1: Export MobileNetV2 features to ONNX (for pet/vehicle recognition) ---
FROM python:3.11-slim AS model-export
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libxcb1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision onnx

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

# --- Stage 1b: Export MobileNetV2 features INT8 TFLite for Coral Edge TPU ---
FROM python:3.11-slim AS coral-model-export
RUN apt-get update && apt-get install -y --no-install-recommends curl gnupg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir tensorflow numpy

# Install Edge TPU compiler
RUN curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && apt-get install -y --no-install-recommends edgetpu-compiler && \
    rm -rf /var/lib/apt/lists/*

# Export MobileNetV2 feature extractor (1280-dim, no classifier head) → INT8 TFLite
RUN mkdir -p /models && python3 -c "\
import tensorflow as tf; \
import numpy as np; \
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet'); \
print(f'MobileNetV2 feature extractor: input {model.input_shape} → output {model.output_shape}'); \
def representative_data_gen(): \
    for _ in range(200): \
        yield [np.random.uniform(0.0, 1.0, (1, 224, 224, 3)).astype(np.float32)]; \
converter = tf.lite.TFLiteConverter.from_keras_model(model); \
converter.optimizations = [tf.lite.Optimize.DEFAULT]; \
converter.representative_dataset = representative_data_gen; \
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]; \
converter.inference_input_type = tf.uint8; \
converter.inference_output_type = tf.uint8; \
tflite_model = converter.convert(); \
with open('/models/mobilenetv2_features_int8.tflite', 'wb') as f: \
    f.write(tflite_model); \
print(f'INT8 TFLite model: {len(tflite_model)} bytes ({len(tflite_model)/1024/1024:.1f} MB)')"

# Compile for Edge TPU (rewrites graph to use TPU custom ops)
RUN edgetpu_compiler -s -o /models/ /models/mobilenetv2_features_int8.tflite && \
    ls -la /models/ && \
    echo "Edge TPU compilation complete"

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

# Coral Edge TPU runtime (feranick's libedgetpu — works on all platforms)
RUN curl -fsSL https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-v2/edgetpu_runtime_20250122_amd64.tar.gz | \
    tar -xz -C /tmp && \
    dpkg -i /tmp/edgetpu_runtime/libedgetpu1-std_16.0tf2.17.1-1_amd64.deb || true && \
    apt-get -f install -y && \
    rm -rf /tmp/edgetpu_runtime

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt ai-edge-litert

# Copy models from build stage (MobileNetV2 for pet/vehicle CNN recognition)
RUN mkdir -p /models-builtin /models
COPY --from=model-export /models/mobilenetv2_features.onnx /models-builtin/mobilenetv2_features.onnx

# Copy Coral Edge TPU model (MobileNetV2 features INT8, compiled for Edge TPU)
COPY --from=coral-model-export /models/mobilenetv2_features_int8_edgetpu.tflite /models-builtin/mobilenetv2_features_edgetpu.tflite

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
