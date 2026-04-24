#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
SEED_PACK="${SEED_PACK:-final_demo}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[1/5] Starting local ChangeGuard server..."
python -c "from changeguard.server.app import run_local_server; run_local_server(host='127.0.0.1', port=8080)" &
SERVER_PID=$!
sleep 1

echo "[2/5] Baseline evaluation on fixed seeds (${SEED_PACK})"
python -m changeguard.training.evaluate_policy --base-url "$BASE_URL" --seed-pack "$SEED_PACK" --candidate baseline

echo "[3/5] Short training run (dry-run by default)"
python -m changeguard.training.train_grpo --base-url "$BASE_URL" --seed-pack short_train --dry-run

echo "[4/5] Optional real GRPO run (set REAL_TRAIN=1 and install trl,datasets)"
if [[ "${REAL_TRAIN:-0}" == "1" ]]; then
  python -m changeguard.training.train_grpo --base-url "$BASE_URL" --seed-pack short_train --no-dry-run --max-steps 8 --prompt-repeats 8
else
  echo "REAL_TRAIN not enabled; skipping --no-dry-run"
fi

echo "[5/5] Candidate evaluation on same fixed seeds (${SEED_PACK})"
python -m changeguard.training.evaluate_policy --base-url "$BASE_URL" --seed-pack "$SEED_PACK" --candidate trained_like

echo "Done."
