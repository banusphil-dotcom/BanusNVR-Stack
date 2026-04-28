#!/bin/bash
# =============================================================================
# BanusNas — Entrypoint: auto-copy built-in models to the /models/ volume
# =============================================================================
set -e

# Copy any models from the image's /models-builtin/ to the bind-mounted /models/
# Only copies regular files that don't already exist. Directories (e.g.
# insightface, person_reid) are handled by the dedicated -r blocks below.
if [ -d /models-builtin ]; then
    for src in /models-builtin/*; do
        # Skip directories — handled separately below with `cp -r`.
        [ -f "$src" ] || continue
        fname=$(basename "$src")
        dest="/models/$fname"
        if [ ! -f "$dest" ] || { [ "$(stat -c%s "$src" 2>/dev/null || echo 0)" -gt 100 ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" -lt 100 ]; }; then
            cp "$src" "$dest"
            echo "Copied built-in model: $fname → /models/"
        fi
    done
fi

# Also ensure InsightFace models are in the volume
if [ -d /models-builtin/insightface ] && [ ! -d /models/insightface ]; then
    cp -r /models-builtin/insightface /models/insightface
    echo "Copied InsightFace models → /models/insightface/"
fi

# Also ensure person ReID model is in the volume
if [ -d /models-builtin/person_reid ] && [ ! -d /models/person_reid ]; then
    cp -r /models-builtin/person_reid /models/person_reid
    echo "Copied person ReID model → /models/person_reid/"
fi

# Copy Coral Edge TPU model if present and not already in volume
coral_src="/models-builtin/mobilenetv2_features_edgetpu.tflite"
coral_dst="/models/mobilenetv2_features_edgetpu.tflite"
if [ -f "$coral_src" ] && { [ ! -f "$coral_dst" ] || [ "$(stat -c%s "$coral_src" 2>/dev/null || echo 0)" -gt "$(stat -c%s "$coral_dst" 2>/dev/null || echo 0)" ]; }; then
    cp "$coral_src" "$coral_dst"
    echo "Copied Coral Edge TPU model → $coral_dst"
fi

# Execute the main command (uvicorn)
exec "$@"
