#!/usr/bin/env bash

# pipeline to run: 
# 1) explainCnn_pretrain.py
# 2) explainCnn/runstreme.sh
# 3) explainCnn/runtomtom.sh

set -Eeuo pipefail
IFS=$'\n\t'

# Config
PY_SCRIPT="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/explainCnn_pretrain.py"
STREME_SH="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/runstreme.sh"
TOMTOM_SH="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/runtomtom.sh"

LOG_DIR="/s/chromatin/m/nobackup/ahmed/DeepPlant/haoxuan/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP() { date +"%F_%H-%M-%S"; }


# Helpers
abort() {
  echo "[ERROR] $*" >&2
  exit 1
}

trap 'echo "[ERROR] Pipeline failed at line ${LINENO}" >&2' ERR

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    abort "No python interpreter found in PATH"
  fi
}

run_and_log() {
  local cmd_desc="$1"; shift
  local logfile="$1"; shift
  echo "[INFO] Starting: ${cmd_desc}"
  echo "[INFO] Logging to: ${logfile}"
  set -o pipefail
  { echo "===== START ${cmd_desc} $(date -Is) ====="; "$@"; rc=$?; echo "===== END ${cmd_desc} $(date -Is) (rc=${rc}) ====="; exit ${rc}; } 2>&1 | tee -a "${logfile}"
  set +o pipefail
  echo "[INFO] Done: ${cmd_desc}"
}


# Preflight checks

[[ -f "${PY_SCRIPT}" ]] || abort "Python script not found: ${PY_SCRIPT}"
[[ -f "${STREME_SH}" ]] || abort "runstreme.sh not found: ${STREME_SH}"
[[ -f "${TOMTOM_SH}" ]] || abort "runtomtom.sh not found: ${TOMTOM_SH}"

PYBIN=$(pick_python)
export PYTHONUNBUFFERED=1


# Step 1: Run explainCnn_pretrain.py
LOG1="${LOG_DIR}/01_explainCnn_pretrain_$(TIMESTAMP).log"
run_and_log "Run explainCnn_pretrain.py" "${LOG1}" \
  "${PYBIN}" -u "${PY_SCRIPT}"


# Step 2: Run STREME steps (runstreme.sh)

LOG2="${LOG_DIR}/02_runstreme_$(TIMESTAMP).log"
run_and_log "Run STREME (runstreme.sh)" "${LOG2}" \
  bash "${STREME_SH}"


# Step 3: Run TOMTOM steps (runtomtom.sh)

LOG3="${LOG_DIR}/03_runtomtom_$(TIMESTAMP).log"
run_and_log "Run TOMTOM (runtomtom.sh)" "${LOG3}" \
  bash "${TOMTOM_SH}"

echo "[SUCCESS] Pipeline completed successfully. Logs in: ${LOG_DIR}"
