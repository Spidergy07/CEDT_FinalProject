#!/usr/bin/env bash
set -euo pipefail

# Simple deploy/run helper for CEDT_FinalProject
# Usage: ./deploy.sh [start|stop|restart|status|logs]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
BACKEND_DIR="$ROOT_DIR/backend"
LOG_FILE="$BACKEND_DIR/server.log"
PID_FILE="$BACKEND_DIR/server.pid"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command '$1' not found in PATH" >&2
    exit 1
  fi
}

ensure_env() {
  if [[ ! -f "$BACKEND_DIR/.env" ]]; then
    if [[ -f "$BACKEND_DIR/env.example" ]]; then
      cp "$BACKEND_DIR/env.example" "$BACKEND_DIR/.env"
      echo "Created backend/.env from env.example. Please edit it with real API keys (CO_API_KEY, GEMINI_API_KEY)."
    else
      cat > "$BACKEND_DIR/.env" <<EOF
# Fill with your real keys before starting
CO_API_KEY=
GEMINI_API_KEY=
# Optional: change port
PORT=3000
EOF
      echo "Created a blank backend/.env. Please fill API keys before starting."
    fi
  fi

  # Basic check that keys exist and are non-empty
  # shellcheck disable=SC1091
  set +u
  source "$BACKEND_DIR/.env" 2>/dev/null || true
  set -u
  if [[ -z "${CO_API_KEY:-}" || -z "${GEMINI_API_KEY:-}" ]]; then
    echo "Warning: CO_API_KEY or GEMINI_API_KEY not set in backend/.env. The server will start but requests will fail." >&2
  fi
}

ensure_node_modules() {
  pushd "$BACKEND_DIR" >/dev/null
  if [[ ! -d node_modules ]]; then
    if [[ -f package-lock.json ]]; then
      npm ci
    else
      npm install
    fi
  fi
  popd >/dev/null
}

ensure_data() {
  local emb="$BACKEND_DIR/pdf_image_embeddings.json"
  local paths="$BACKEND_DIR/processed_image_paths.txt"
  if [[ ! -f "$emb" || ! -f "$paths" ]]; then
    echo "Warning: Missing embeddings or paths file." >&2
    echo "  Expected: $emb and $paths" >&2
    echo "  The API may fail to serve search/answer endpoints without these." >&2
  fi
}

start() {
  ensure_env
  ensure_node_modules
  ensure_data

  if [[ -f "$PID_FILE" ]]; then
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "Server already running with PID $(cat "$PID_FILE")."
      exit 0
    else
      rm -f "$PID_FILE"
    fi
  fi

  pushd "$BACKEND_DIR" >/dev/null
  # Use .env PORT if provided, otherwise api.js defaults to 3000
  echo "Starting backend (logging to $LOG_FILE)..."
  nohup npm run start >"$LOG_FILE" 2>&1 & echo $! >"$PID_FILE"
  popd >/dev/null

  sleep 1
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    local port_display
    # shellcheck disable=SC1091
    set +u; source "$BACKEND_DIR/.env" 2>/dev/null || true; set -u
    port_display="${PORT:-3000}"
    echo "Server running: http://localhost:$port_display"
    echo "Frontend available at the same URL."
  else
    echo "Failed to start server. Check logs: $LOG_FILE" >&2
    exit 1
  fi
}

stop() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping server PID $pid..."
      kill "$pid" || true
      # Wait briefly, then force if needed
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" || true
      fi
    fi
    rm -f "$PID_FILE"
    echo "Stopped."
  else
    echo "No PID file found; server not running."
  fi
}

status() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    local port_display
    # shellcheck disable=SC1091
    set +u; source "$BACKEND_DIR/.env" 2>/dev/null || true; set -u
    port_display="${PORT:-3000}"
    echo "Server is running (PID $(cat "$PID_FILE")) on port $port_display."
  else
    echo "Server is not running."
    return 1
  fi
}

logs() {
  if [[ -f "$LOG_FILE" ]]; then
    echo "Tailing logs (Ctrl+C to stop)..."
    tail -n 200 -f "$LOG_FILE"
  else
    echo "No log file at $LOG_FILE yet."
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  restart) stop || true; start ;;
  status) status ;;
  logs) logs ;;
  *) echo "Usage: $0 [start|stop|restart|status|logs]"; exit 2 ;;
esac

exit 0

