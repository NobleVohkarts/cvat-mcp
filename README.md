# cvat-mcp

An MCP server that lets an AI assistant review annotations in [CVAT](https://github.com/cvat-ai/cvat):
browse tasks, look at frames with their annotations, and leave issues for annotators.

## Why

Annotation review is mostly looking: is every object boxed, is every label right, does every box
fit its object? An assistant with vision can do a first pass over a task and leave issues where a
human reviewer should look. The server **gives the model eyes**: it renders frames with the
annotations drawn on them and returns the exact objects. **The model does the judging** and
reports each problem as a CVAT issue on the right frame of the right job.

## Tools

| Tool | Modes | Use it to |
| --- | --- | --- |
| `server_info` | both | See CVAT and SDK versions, compatibility, the current user, the mode and the available tools. Call first or when something fails. |
| `list_tasks` | both | Find tasks (newest first, `limit` ≤ 100) with frame count, status, job progress and the `total` of all matches. |
| `get_task_overview` | both | Understand one task: labels, jobs with frame ranges, stage, state, assignee, objects per label. |
| `render_frame` | both | Look at a frame: JPEG with every box and a semi-transparent `#id label` caption drawn, plus a legend with exact boxes in original pixels. `include_clean=True` adds the same frame without annotations, for crowded frames. `job_id` draws one job's annotations (e.g. ground truth). |
| `get_frame_annotations` | both | Get exact data for one frame's objects and tags: id, label, type, box (plus points for polygons, polylines and points), attributes, source. `job_id` reads one job's annotations (e.g. ground truth). |
| `download_annotations` | both | Read all annotations of a job (or of every annotation job of a task) as one compact JSON per job, returned as an `application/json` embedded resource. Refuses above 2000 shapes. |
| `create_report` | both | Measure a task's annotation jobs against its ground-truth job (CVAT quality control): waits for a fresh report, returns counts, scores and conflicts by frame with the ids involved; needs a token that can write. |
| `list_issues` | both | See the issues already filed on a task or one frame: message, position, author, resolved, replies, and `own` (filed under the token's CVAT account). |
| `create_issue` | **write** | Report one problem to the annotator: an issue on a frame of a job, at an object's box or a given region. |
| `delete_issue` | **write** | Remove one of the token account's own issues (with its comments), e.g. to rewrite issues: list → create the new → delete the old. |
| `create_annotations` | **write** | Add missing objects or tags (rectangles by box, polygons/polylines/points by points), all checked first, then added in one step. `job_id` writes into one job, e.g. ground truth. |
| `update_annotations` | **write** | Correct existing objects or tags by id: label, box or points, attributes, occluded; only what changes is given. `job_id` edits one job's objects. |
| `delete_annotations` | **write** | Remove objects or tags by explicit ids (shape and tag ids are separate lists), all or none. `job_id` deletes from one job. |
| `import_annotations` | **write** | Load an annotation file in any CVAT import format (default `COCO 1.0`) into a job (e.g. ground truth) or a task, from one file in a CVAT cloud storage (`cloud_storage_id` + `cloud_path`; CVAT reads it through the storage); appends unless `mode="replace"`; returns the object counts. |
| `export_dataset` | **write** | Save a task as a dataset zip in a CVAT format (e.g. `COCO 1.0`) into a CVAT cloud storage (`cloud_storage_id` + `cloud_path`; CVAT uploads it); waits for CVAT's background export; refuses an existing object unless `overwrite=true`. |
| `auto_annotate` | **write** (with the extra) | Run a built-in torchvision model from the CVAT SDK (`output`: boxes, polygons, masks, skeletons or tags; default models `fasterrcnn_resnet50_fpn_v2`, `maskrcnn_resnet50_fpn_v2`, `keypointrcnn_resnet50_fpn`, `resnet50`; the `model` argument lists the alternatives) on one frame (waits, returns the new objects) or on a whole task (background run, saved every 10 frames, so a failed run keeps what it saved). Matches labels by name and counts the skipped ones; appends unless `mode="replace"` (shapes and tags, never tracks; each batch's old ones are deleted after its new ones are saved); one run at a time; tasks with overlapping jobs are refused. |
| `get_auto_annotation_run` | **write** (with the extra) | Progress of the latest background `auto_annotate` run (only that one is kept; an older `run_id` gets it too): state, frames saved, objects saved per label, error. In memory only; a restart forgets it and stops it if it is running (the frames it saved stay). |
| `create_task` | **write** | Create a task from image URLs or from a CVAT cloud storage (`cloud_storage_id` + `cloud_path`: a folder, glob or object), with labels or a project; waits for CVAT to process the data and leaves nothing behind if the upload fails. |
| `update_task` | **write** | Rename a task, assign it, move it into a project or add labels (never remove them). |
| `delete_task` | **write** | Delete a task owned by the token's CVAT account, with all its jobs, annotations and issues. |
| `list_cloud_storages` | read | The cloud storages (buckets) CVAT knows: bucket, endpoint, prefix, owner and whether it has keys; never the keys. |
| `create_cloud_storage` | **write** | Register an S3-compatible bucket (AWS S3, or any S3 API endpoint such as Cloudflare R2 or MinIO) by bucket, endpoint and region, with keys or anonymously; CVAT checks it can reach it; an exact repeat is refused. |
| `update_cloud_storage` | **write** | Rename a cloud storage or replace its keys. |
| `delete_cloud_storage` | **write** | Delete a cloud storage owned by the token's account that no task's frames come from (the bucket is untouched). |
| `update_job` | **write** | Assign a job or move it through stages and states (a stage change without a state resets it to "new"). |
| `create_gt_job` | **write** | Add a ground-truth job: random frames (optional seed) or given task frames; one per image task. |
| `delete_gt_job` | **write** | Delete a ground-truth job of a task owned by the token's CVAT account; annotation jobs can't be deleted. |
| `update_quality_settings` | **write** | Switch a task's quality requirements (e.g. `Base rectangle`) on or off where its settings live (its project's or its own) and choose between them, so `create_report` has something to compare; says how many tasks share project settings. |

In write mode the model can fix annotations, and the instructions say to do so only when the
user asks. Each annotation tool checks every item (frame, label and its type, attribute values,
geometry) before it writes, then applies the whole list in one CVAT request, or nothing. Deletes
take explicit ids only; the SDK's `remove_annotations()`, which deletes everything when given an
empty list, is never used. Tracks stay read-only here, and so do tasks with overlapping jobs
unless one job is edited (`job_id`). Ground-truth annotations are reached with `job_id` of the
GT job; the instructions say to edit ground truth only when the user explicitly asks, since
annotators are measured against it. Tasks can be created and edited; `delete_task` and `delete_issue` are limited to
what the token's CVAT account owns, so anyone else's tasks and issues stay untouched. Jobs can be
reassigned and moved through stages; ground-truth jobs can be created, and deleted in the
account's own tasks. Annotation jobs can't be created or deleted: CVAT makes them from the task's
segments. Tasks and issues you created in the UI with that same account count as its own, so
give the server a token of a separate reviewer account if those must be protected too.

## Design

| Principle | Where it lives |
| --- | --- |
| **A tool is a user task, not an endpoint.** Verb + object names; each description says when to use the tool and how it differs from its neighbours. | `tools/*.py` docstrings |
| **Asynchrony stays inside.** Tools wait for CVAT background work themselves. | `export_dataset` returns only when CVAT's background export (`202` + `rq_id`) is done and uploaded to the bucket (`services/exports.py`); `create_report` waits for the quality report request (`services/quality.py`) |
| **Protect the model's context.** `limit` + `total`, summaries instead of raw data, pydantic responses with only the needed fields, images downscaled to `max_side` and sent as JPEG. | `services/*.py` models, `render.py` |
| **Security.** What isn't in the mode isn't registered; `ToolAnnotations` hints; Personal Access Token only (the server never stores or acts with a password); no local files: data moves through CVAT cloud storages or URLs; the token is never logged or returned. | `server.tool()`; remote access: `auth.py` |
| **The user confirms the biggest changes that can't be undone.** `delete_task`, `delete_gt_job`, `delete_cloud_storage`, `import_annotations` and `auto_annotate` with `mode="replace"`, and `export_dataset` with `overwrite=true` ask the user through the client's confirmation form (MCP elicitation) before touching CVAT. A decline returns "the user declined: …" and changes nothing. Clients that can't show a form run these calls as before; the instructions' "only when the user asks" rules apply either way. | `confirm.py`, the tools' `confirmation` parameter |
| **Errors are written for the model.** 401/403/404/400 and argument mistakes become one-line hints with the next step. mcp shows only `ToolError` text, so every tool is wrapped. | `errors.explain`, `server.tool_errors`, `server.reject_unknown_arguments` |
| **The server gives the model eyes, the model judges.** No heuristic checks on the server; rendering and facts only. | `render.py`, `services/annotations.py` |

## Architecture

```
cvat-mcp/
├── src/cvat_mcp/
│   ├── __main__.py      # entry point: logging, config errors as one line, stdio or HTTP
│   ├── auth.py          # OAuth for remote clients: CVAT login page, in-memory tokens
│   ├── server.py        # MCPServer, lifespan (one SDK client, token checked at startup),
│   │                    # instructions, tool() registration by mode, tool_errors wrapper
│   ├── confirm.py       # confirmation forms (elicitation) for calls that can't be undone
│   ├── config.py        # settings from env/.env (pydantic-settings)
│   ├── errors.py        # exception -> message for the model (no mcp dependency)
│   ├── render.py        # draws annotations over a frame, downscales, encodes JPEG
│   ├── services/        # logic on cvat-sdk, pydantic results, no mcp dependency
│   └── tools/           # thin MCP wrappers: descriptions and argument types
├── scripts/             # serve-public.sh (tunnel + HTTP), check_remote.py (OAuth smoke test)
└── tests/               # pytest, no server needed
```

A call goes `tool (tools/) -> service (services/ or render.py) -> cvat-sdk -> CVAT`. Services
and `render.py` don't import mcp (a test enforces it), so the same logic can back a CLI or a
different protocol.

## Install and configure

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in CVAT_ACCESS_TOKEN
```

Optional: `uv sync --extra auto-annotation` adds CPU builds of torch and torchvision (about 1 GB)
and, in write mode, the `auto_annotate` and `get_auto_annotation_run` tools. Without the extra
those two tools don't exist. `uv run` keeps the extra; a later plain `uv sync` removes it again.
Each model downloads its weights (about 170 MB) into `~/.cache/torch` on first use.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CVAT_URL` | `http://localhost:8080` | CVAT server URL |
| `CVAT_ACCESS_TOKEN` | — (required) | Personal Access Token (CVAT UI → profile → Access tokens). Write mode needs a token **without** "read only". |
| `CVAT_MCP_MODE` | `read-only` | `read-only` or `write` (adds the write tools for issues, annotations, tasks and jobs) |
| `CVAT_MCP_TRANSPORT` | `stdio` | `http` serves Streamable HTTP with OAuth (set by `scripts/serve-public.sh`) |
| `CVAT_MCP_PUBLIC_URL` | — | Public `https` URL in HTTP mode (the tunnel's or proxy's) |
| `CVAT_MCP_PORT` | `8765` | Local port in HTTP mode; bound to `127.0.0.1` only |
| `CVAT_MCP_REDIRECT_HOSTS` | `claude.ai,claude.com,localhost,127.0.0.1` | Hosts OAuth clients may register as callbacks |

`.env` is read from the **current working directory**. That's why the client commands below pass
`--directory`.

Run it directly (stdio): `uv run cvat-mcp`. A bad token, an unreachable URL or a missing variable
stops it with one line saying what to fix.

## Use with Claude Code

```bash
claude mcp add cvat --scope local -e CVAT_MCP_MODE=write -- \
  uv run --directory /absolute/path/to/cvat-mcp cvat-mcp
claude mcp list   # cvat: ... ✔ Connected
```

### Claude Desktop

Claude Desktop starts the same command as Claude Code above:

```json
{
  "mcpServers": {
    "cvat": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/cvat-mcp", "cvat-mcp"],
      "env": { "CVAT_MCP_MODE": "write" }
    }
  }
}
```

## Use from Claude on the web

claude.ai connects to MCP servers from Anthropic's cloud, so the server needs a public HTTPS
address. It serves HTTP mode on `127.0.0.1` only, protected by OAuth that it implements itself,
and a tunnel or reverse proxy on the same machine gives it the public address.

To try it, `scripts/serve-public.sh` starts a Cloudflare **quick tunnel** (no account needed) and
the server, with a new URL on every run:

```bash
CVAT_MCP_MODE=write scripts/serve-public.sh   # or read-only (the .env default)
# Add this connector URL in claude.ai: https://<random>.trycloudflare.com/mcp
# Restart only the server (same URL): kill -USR1 <pid>
```

For a fixed address, point a reverse proxy (nginx, Caddy, Traefik) or a named Cloudflare tunnel
at `http://127.0.0.1:8765` (`CVAT_MCP_PORT`) and start the server with its public URL:

```bash
CVAT_MCP_TRANSPORT=http CVAT_MCP_PUBLIC_URL=https://mcp.example.com CVAT_MCP_MODE=write \
  uv run cvat-mcp
```

The proxy must send `X-Forwarded-For` (Caddy, Traefik and Cloudflare tunnels do by default), so
failed logins count per client, and must not buffer the streamed answers. With a second proxy in
front of it (e.g. Cloudflare's in front of nginx), the inner one must restore the client address
first (nginx: `set_real_ip_from` + `real_ip_header`), or all clients share one lockout; the server
logs a warning when a login comes without `X-Forwarded-For`. For nginx:

```nginx
location / {
    proxy_pass http://127.0.0.1:8765;
    proxy_http_version 1.1;
    proxy_set_header Host $http_host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_buffering off;
    proxy_read_timeout 1h;  # long tool calls and idle streams
}
```

In claude.ai: **Settings → Connectors → Add custom connector**, paste the URL with `/mcp` at the
end, **Connect**, then sign in on the server's page with the CVAT account that owns
`CVAT_ACCESS_TOKEN`.

How it is protected:

- **OAuth inside the server** (`auth.py`): claude.ai registers itself, sends you to `/login`, and
  gets short-lived tokens (1 h access, 30-day refresh, revocable).
- **Only the token's owner can log in.** The page checks the CVAT username and password against
  CVAT once and accepts only the user who owns the server's access token, so another CVAT user
  can't borrow its rights. The password is never stored or logged.
- **Callbacks only to claude.ai / claude.com** (and localhost for testing): other clients can't
  register.
- **Brute force:** about 1 s delay per failed login; 10 failures from one client address (the
  last one the local proxy added to `X-Forwarded-For`, so a client can't fake it) lock it out for
  10 minutes.
- **Strict CSP** on the login page: the form posts only to the server and to the client's callback.

Agent says a tool lacks a documented option? claude.ai has a stale tool list: remove and re-add
the connector (a refresh isn't enough). `server_info` reports `cvat_mcp_version`.

Limits: tokens live in memory, so after a server restart claude.ai asks you to **Connect** again
(the `kill -USR1` restart keeps the quick-tunnel URL). `uv run python scripts/check_remote.py <url>` runs the whole
OAuth flow and a tool call against a running server (it reads `CVAT_UI_USERNAME` /
`CVAT_UI_PASSWORD` from `.env`; the server never does).

## Try it in MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run cvat-mcp            # browser UI
npx @modelcontextprotocol/inspector --cli -e CVAT_MCP_MODE=write uv run cvat-mcp --method tools/list
```

The Inspector starts the server with a minimal environment, so pass variables with `-e`, not by
exporting them.

## Development

```bash
uv run pytest
uv run pre-commit run --all-files
# optional git hook: uv run pre-commit install
```

Style follows CVAT: black and isort (line length 100), pylint with `.pylintrc` (copied from the CVAT repo).
