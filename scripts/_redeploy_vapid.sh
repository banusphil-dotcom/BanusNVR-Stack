#!/bin/bash
set -e
echo Twins@2021!! | sudo -S bash -c '
set -e
echo "=== Pull new API image ==="
cd /volume2/docker/nvr
docker compose pull banusnvr-api 2>&1 | tail -10

echo "=== Delete broken VAPID key row (forces regeneration on next startup) ==="
docker exec banusnvr-db psql -U banusnvr -d banusnvr -c "DELETE FROM system_settings WHERE key = '"'"'vapid_keys'"'"';"

echo "=== Recreate API container ==="
docker compose up -d banusnvr-api 2>&1 | tail -10

echo "=== Wait for startup ==="
sleep 12

echo "=== Verify VAPID load ==="
docker logs banusnvr-api 2>&1 | grep -iE "vapid|generating" | tail -10

echo "=== Verify push works (test against py_vapid) ==="
docker cp /tmp/_diag_vapid.py banusnvr-api:/tmp/d.py 2>/dev/null || true
docker exec -e PYTHONPATH=/app -w /app banusnvr-api python /tmp/d.py 2>&1 | tail -15
'
