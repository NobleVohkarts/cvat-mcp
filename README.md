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

| Tool | Mode | Use it to |
| --- | --- | --- |
| `server_info` | read-only | See CVAT and SDK versions, compatibility, the current user, the mode and the available tools. Call first or when something fails. |
| `list_tasks` | read-only | Find tasks (newest first, `limit` ≤ 100) with frame count, status, job progress and the `total` of all matches. |
| `get_task_overview` | read-only | Understand one task: labels, jobs with frame ranges, stage, state, assignee, objects per label. |
| `render_frame` | read-only | Look at a frame: JPEG with every box and `#id label` caption drawn, plus a legend with exact boxes in original pixels. |
| `get_frame_annotations` | read-only | Get exact data for one frame's objects: id, label, type, box, attributes, source. |
| `create_issue` | **write** | Report one problem to the annotator: an issue on a frame of a job, at an object's box or a given region. |

Deleting or changing tasks, jobs and annotations is deliberately not available. The server's
instructions tell the model so, and it points the user to the CVAT UI.

## Design

| Principle | Where it lives |
| --- | --- |
| **A tool is a user task, not an endpoint.** Verb + object names; each description says when to use the tool and how it differs from its neighbours. | `tools/*.py` docstrings |
| **Asynchrony stays inside.** Tools wait for CVAT background work themselves. | cvat-sdk polls imports/exports; today's tools make synchronous requests |
| **Protect the model's context.** `limit` + `total`, summaries instead of raw data, pydantic responses with only the needed fields, images downscaled to `max_side` and sent as JPEG. | `services/*.py` models, `render.py` |
| **Security.** What isn't in the mode isn't registered; `ToolAnnotations` hints; Personal Access Token only (no password setting); file paths only inside `CVAT_MCP_WORKDIR`; the token is never logged or returned. | `server.tool()`, `config.py` (`resolve_in_workdir`) |
| **Errors are written for the model.** 401/403/404/400 and argument mistakes become one-line hints with the next step. mcp shows only `ToolError` text, so every tool is wrapped. | `errors.explain`, `server.tool_errors` |
| **The server gives the model eyes, the model judges.** No heuristic checks on the server; rendering and facts only. | `render.py`, `services/annotations.py` |

## Architecture

```
cvat-mcp/
├── src/cvat_mcp/
│   ├── __main__.py      # entry point: logging, config errors as one line, stdio transport
│   ├── server.py        # MCPServer, lifespan (one SDK client, token checked at startup),
│   │                    # instructions, tool() registration by mode, tool_errors wrapper
│   ├── config.py        # settings from env/.env (pydantic-settings), workdir path sandbox
│   ├── errors.py        # exception -> message for the model (no mcp dependency)
│   ├── render.py        # draws annotations over a frame, downscales, encodes JPEG
│   ├── services/        # logic on cvat-sdk, pydantic results, no mcp dependency
│   └── tools/           # thin MCP wrappers: descriptions and argument types
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
| `CVAT_ACCESS_TOKEN` | — (required) | Personal Access Token (CVAT UI → profile → Access tokens). Write mode needs a token **without** "read only". Passwords are deliberately not supported. |
| `CVAT_MCP_MODE` | `read-only` | `read-only` or `write` (adds `create_issue`) |
| `CVAT_MCP_WORKDIR` | `workdir` | The only directory the server may read or write files in |

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

- `create_task`, `import_annotations`, `update_job`;
- a progress adapter for long operations;
- a `full` mode with confirmation via elicitation;
- a `review_task` prompt;
- protocol and integration tests;
- mypy strict;
- remote mode (Streamable HTTP + OAuth).

Learned while building it:

- **A `region` (zoom) parameter for `render_frame`.** Given a shell, the model wrote its own crop
  script to look closer at small objects.
- **Tracks and frame-level tags in the frame tools.** Tracked objects are counted but not drawn.
- **Per-frame annotation fetching,** once CVAT supports a frame filter.
- **Reading `.env` from the project directory,** not the working directory.
- **`export_dataset(task_id, format, include_images=False)`,** saving through the workdir sandbox.

