#!/usr/bin/env bash
# Serve cvat-mcp to Claude on the web: a Cloudflare quick tunnel + the server in HTTP mode with
# OAuth. The tunnel URL is new on every run of this script; `kill -HUP <pid>` (printed below)
# restarts only the server, keeping the tunnel and its URL. Ctrl-C stops both.
set -euo pipefail

cd "$(dirname "$0")/.."
PORT="${CVAT_MCP_PORT:-8765}"
LOG="$(mktemp -t cvat-mcp-tunnel.XXXXXX)"

cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" >"$LOG" 2>&1 &
TUNNEL_PID=$!
SERVER_PID=""
trap 'kill $SERVER_PID "$TUNNEL_PID" 2>/dev/null; rm -f "$LOG"' EXIT
trap 'exit 130' INT TERM  # run the EXIT trap: stop the server and the tunnel

URL=""
for _ in $(seq 60); do
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
    [ -n "$URL" ] && break
    kill -0 "$TUNNEL_PID" 2>/dev/null || { cat "$LOG" >&2; exit 1; }
    sleep 1
done
[ -n "$URL" ] || { echo "cvat-mcp: no tunnel URL after 60 s" >&2; cat "$LOG" >&2; exit 1; }

echo "Add this connector URL in claude.ai: ${URL}/mcp" >&2
echo "Restart only the server (same URL): kill -HUP $$" >&2

RESTART=0
trap 'RESTART=1; kill "$SERVER_PID" 2>/dev/null' HUP
while true; do
    CVAT_MCP_TRANSPORT=http CVAT_MCP_PUBLIC_URL="$URL" CVAT_MCP_PORT="$PORT" uv run cvat-mcp &
    SERVER_PID=$!
    wait "$SERVER_PID" || true
    # `wait` returns early when the HUP trap fires; let the old server finish exiting.
    while kill -0 "$SERVER_PID" 2>/dev/null; do sleep 0.2; done
    [ "$RESTART" = 1 ] || break
    RESTART=0
    echo "cvat-mcp: restarting the server; claude.ai will ask to Connect again" >&2
done
