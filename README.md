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
| `render_frame` | both | Look at a frame: JPEG with every box and a semi-transparent `#id label` caption drawn, plus a legend with exact boxes in original pixels. `include_clean=True` adds the same frame without annotations, for crowded frames. |
| `get_frame_annotations` | both | Get exact data for one frame's objects and tags: id, label, type, box (plus points for polygons, polylines and points), attributes, source. |
| `export_dataset` | both | Save a task as a dataset file in a CVAT format (e.g. `COCO 1.0`) inside `CVAT_MCP_WORKDIR`; waits for CVAT's background export, never overwrites. |
| `list_issues` | both | See the issues already filed on a task or one frame: message, position, author, resolved, replies, and `own` (filed under the token's CVAT account). |
| `create_issue` | **write** | Report one problem to the annotator: an issue on a frame of a job, at an object's box or a given region. |
| `delete_issue` | **write** | Remove one of the token account's own issues (with its comments), e.g. to rewrite issues: list → create the new → delete the old. |
| `create_annotations` | **write** | Add missing objects or tags (rectangles by box, polygons/polylines/points by points), all checked first, then added in one step. |
| `update_annotations` | **write** | Correct existing objects or tags by id: label, box or points, attributes, occluded; only what changes is given. |
| `delete_annotations` | **write** | Remove objects or tags by explicit ids (shape and tag ids are separate lists), all or none. |
| `create_task` | **write** | Create a task from images in `CVAT_MCP_WORKDIR` or from URLs, with labels or a project; waits for CVAT to process the data and leaves nothing behind if the upload fails. |
| `update_task` | **write** | Rename a task, assign it, move it into a project or add labels (never remove them). |
| `delete_task` | **write** | Delete a task owned by the token's CVAT account, with all its jobs, annotations and issues. |
| `update_job` | **write** | Assign a job or move it through stages and states (a stage change without a state resets it to "new"). |
| `create_gt_job` | **write** | Add a ground-truth job: random frames (optional seed) or given task frames; one per image task. |
| `delete_gt_job` | **write** | Delete a ground-truth job of a task owned by the token's CVAT account; annotation jobs can't be deleted. |

In write mode the model can fix annotations, and the instructions say to do so only when the
user asks. Each annotation tool checks every item (frame, label and its type, attribute values,
geometry) before it writes, then applies the whole list in one CVAT request, or nothing. Deletes
take explicit ids only; the SDK's `remove_annotations()`, which deletes everything when given an
empty list, is never used. Tracks, ground-truth annotations and tasks with overlapping jobs stay
read-only here. Tasks can be created and edited; `delete_task` and `delete_issue` are limited to
what the token's CVAT account owns, so anyone else's tasks and issues stay untouched. Jobs can be
reassigned and moved through stages; ground-truth jobs can be created, and deleted in the
account's own tasks. Annotation jobs can't be created or deleted: CVAT makes them from the task's
segments. Tasks and issues you created in the UI with that same account count as its own, so
give the server a token of a separate reviewer account if those must be protected too.

## Design

| Principle | Where it lives |
| --- | --- |
| **A tool is a user task, not an endpoint.** Verb + object names; each description says when to use the tool and how it differs from its neighbours. | `tools/*.py` docstrings |
| **Asynchrony stays inside.** Tools wait for CVAT background work themselves. | `export_dataset` returns only when CVAT's background export (`202` + `rq_id`) is done and downloaded (`services/exports.py`) |
| **Protect the model's context.** `limit` + `total`, summaries instead of raw data, pydantic responses with only the needed fields, images downscaled to `max_side` and sent as JPEG. | `services/*.py` models, `render.py` |
| **Security.** What isn't in the mode isn't registered; `ToolAnnotations` hints; Personal Access Token only (the server never stores or acts with a password); file paths only inside `CVAT_MCP_WORKDIR`; the token is never logged or returned. | `server.tool()`; `config.resolve_in_workdir` (used by `export_dataset`); remote access: `auth.py` |
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
│   ├── config.py        # settings from env/.env (pydantic-settings), workdir path sandbox
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
cd cvat-mcp
uv sync
cp .env.example .env   # then fill in CVAT_ACCESS_TOKEN
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CVAT_URL` | `http://localhost:8080` | CVAT server URL |
| `CVAT_ACCESS_TOKEN` | — (required) | Personal Access Token (CVAT UI → profile → Access tokens). Write mode needs a token **without** "read only". |
| `CVAT_MCP_MODE` | `read-only` | `read-only` or `write` (adds the write tools for issues, annotations, tasks and jobs) |
| `CVAT_MCP_WORKDIR` | `workdir` | The only directory the server may read or write files in |
| `CVAT_MCP_TRANSPORT` | `stdio` | `http` serves Streamable HTTP with OAuth (set by `scripts/serve-public.sh`) |
| `CVAT_MCP_PUBLIC_URL` | — | Public `https` URL in HTTP mode (the tunnel URL) |
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

### Claude Desktop (not tested)

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
address. `scripts/serve-public.sh` starts a Cloudflare **quick tunnel** (no account needed) and the
server in HTTP mode, protected by OAuth that the server implements itself:

```bash
cd cvat-mcp
CVAT_MCP_MODE=write scripts/serve-public.sh   # or read-only (the .env default)
# Add this connector URL in claude.ai: https://<random>.trycloudflare.com/mcp
# Restart only the server (same URL): kill -USR1 <pid>
```

In claude.ai: **Settings → Connectors → Add custom connector**, paste the URL, **Connect**, then
sign in on the server's page with the CVAT account that owns `CVAT_ACCESS_TOKEN`.

How it is protected:

- **OAuth inside the server** (`auth.py`): claude.ai registers itself, sends you to `/login`, and
  gets short-lived tokens (1 h access, 30-day refresh, revocable).
- **Only the token's owner can log in.** The page checks the CVAT username and password against
  CVAT once and accepts only the user who owns the server's access token, so another CVAT user
  can't borrow its rights. The password is never stored or logged.
- **Callbacks only to claude.ai / claude.com** (and localhost for testing): other clients can't
  register.
- **Brute force:** about 1 s delay per failed login; 10 failures lock logins for 10 minutes.
- **Bound to 127.0.0.1;** the tunnel is the only way in. The login page sends a strict CSP that
  allows the form to post to the server and to the client's callback only.

Agent says a tool lacks a documented option? claude.ai has a stale tool list: remove and re-add
the connector (a refresh isn't enough). `server_info` reports `cvat_mcp_version`.

Limits: the quick-tunnel URL is new every time the script starts (re-add the connector then);
tokens live in memory, so after a server restart claude.ai asks you to **Connect** again (the
`kill -USR1` restart keeps the URL). `uv run python scripts/check_remote.py <url>` runs the whole
OAuth flow and a tool call against a running server (it reads `CVAT_UI_USERNAME` /
`CVAT_UI_PASSWORD` from `.env`; the server never does).

## Try it in MCP Inspector

```bash
cd cvat-mcp
npx @modelcontextprotocol/inspector uv run cvat-mcp            # browser UI
npx @modelcontextprotocol/inspector --cli -e CVAT_MCP_MODE=write uv run cvat-mcp --method tools/list
```

The Inspector starts the server with a minimal environment, so pass variables with `-e`, not by
exporting them.

## Development

```bash
cd cvat-mcp
uv run pytest
cd .. && uv run --project cvat-mcp pre-commit run --config cvat-mcp/.pre-commit-config.yaml --all-files
# optional git hook: uv run --project cvat-mcp pre-commit install --config cvat-mcp/.pre-commit-config.yaml
```

Style follows the monorepo: black and isort (line length 100), pylint with the root `.pylintrc`.

## Roadmap

Not done yet:

- `import_annotations`;
- a progress adapter for long operations;
- a `full` mode with confirmation via elicitation;
- a `review_task` prompt;
- protocol and integration tests;
- mypy strict;
- a stable public URL (named Cloudflare tunnel on your own domain) and tokens that survive restarts.

Learned while building it:

- **A `region` (zoom) parameter for `render_frame`.** Given a shell, the model wrote its own crop
  script to look closer at small objects. In very crowded frames (35+ captions) it can miss a
  wrong label.
- **Tracks in the frame tools.** Tracked objects are counted but not drawn or edited.
- **Per-frame annotation fetching,** once CVAT supports a frame filter.
- **Reading `.env` from the project directory,** not the working directory.

