#!/usr/bin/env bash
# Serve cvat-mcp to Claude on the web: a Cloudflare quick tunnel + the server in HTTP mode with
# OAuth. The tunnel URL is new on every run, so add the connector in claude.ai again each time.
set -euo pipefail

cd "$(dirname "$0")/.."
PORT="${CVAT_MCP_PORT:-8765}"
LOG="$(mktemp -t cvat-mcp-tunnel.XXXXXX)"

cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" >"$LOG" 2>&1 &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2>/dev/null; rm -f "$LOG"' EXIT

URL=""
for _ in $(seq 60); do
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
    [ -n "$URL" ] && break
    kill -0 "$TUNNEL_PID" 2>/dev/null || { cat "$LOG" >&2; exit 1; }
    sleep 1
done
[ -n "$URL" ] || { echo "cvat-mcp: no tunnel URL after 60 s" >&2; cat "$LOG" >&2; exit 1; }

echo "Add this connector URL in claude.ai: ${URL}/mcp" >&2
CVAT_MCP_TRANSPORT=http CVAT_MCP_PUBLIC_URL="$URL" CVAT_MCP_PORT="$PORT" uv run cvat-mcp
