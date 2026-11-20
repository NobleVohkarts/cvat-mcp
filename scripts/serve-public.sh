#!/usr/bin/env bash
# Serve cvat-mcp to Claude on the web: a Cloudflare quick tunnel + the server in HTTP mode with
# OAuth. The tunnel URL is new on every run of this script; `kill -USR1 <pid>` (printed below)
# restarts only the server, keeping the tunnel and its URL. Ctrl-C, TERM or HUP stop both.
set -euo pipefail

cd "$(dirname "$0")/.."
# the port: shell env first, then .env, then the default (the server reads the same order)
PORT="${CVAT_MCP_PORT:-$(sed -n 's/^CVAT_MCP_PORT=//p' .env 2>/dev/null | tail -1)}"
PORT="${PORT:-8765}"
LOG="$(mktemp -t cvat-mcp-tunnel.XXXXXX)"

cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}" >"$LOG" 2>&1 &
TUNNEL_PID=$!
SERVER_PID=""
stop_server() {  # the whole group (uv + python); force it if open connections keep it alive
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || return 0
    for _ in $(seq 50); do
        kill -0 -- "-$SERVER_PID" 2>/dev/null || return 0
        sleep 0.1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
}
trap 'stop_server; kill "$TUNNEL_PID" 2>/dev/null; rm -f "$LOG"' EXIT
trap 'exit 130' INT TERM HUP  # run the EXIT trap: stop the server and the tunnel

URL=""
for _ in $(seq 60); do
    # not cloudflared's own API host, which shows up in its error lines
    URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | grep -v '//api\.' | head -1 || true)"
    [ -n "$URL" ] && break
    kill -0 "$TUNNEL_PID" 2>/dev/null || { cat "$LOG" >&2; exit 1; }
    sleep 1
done
[ -n "$URL" ] || { echo "cvat-mcp: no tunnel URL after 60 s" >&2; cat "$LOG" >&2; exit 1; }

echo "Add this connector URL in claude.ai: ${URL}/mcp" >&2
echo "Restart only the server (same URL): kill -USR1 $$" >&2

RESTART=0
trap 'RESTART=1' USR1
while true; do
    # own process group, so stop_server reaches python behind uv
    CVAT_MCP_TRANSPORT=http CVAT_MCP_PUBLIC_URL="$URL" CVAT_MCP_PORT="$PORT" \
        setsid uv run cvat-mcp &
    SERVER_PID=$!
    RESTART=0
    wait "$SERVER_PID" || true  # a USR1 interrupts this wait
    [ "$RESTART" = 1 ] || break  # the server exited on its own: stop everything
    stop_server
    echo "cvat-mcp: restarting the server; claude.ai will ask to Connect again" >&2
done
