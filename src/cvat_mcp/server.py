import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from cvat_sdk import Client, make_client
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from urllib3.exceptions import HTTPError

from cvat_mcp.config import Settings
from cvat_mcp.errors import F, tool_errors

logger = logging.getLogger("cvat_mcp")

settings = Settings()


def _connect(s: Settings) -> Client:
    # make_client() also checks the server version and logs a warning if it is unsupported.
    try:
        client = make_client(s.url, access_token=s.access_token.get_secret_value())
        user = client.users.retrieve_current_user()
    except HTTPError:
        sys.exit(f"cvat-mcp: cannot reach CVAT at {s.url}. Is the server running? Check CVAT_URL.")
    except ApiException as e:
        if e.status in (401, 403):
            sys.exit(
                "cvat-mcp: CVAT rejected the token. Create a new Personal Access Token in the "
                "CVAT UI and put it into CVAT_ACCESS_TOKEN."
            )
        sys.exit(f"cvat-mcp: CVAT at {s.url} answered {e.status} {e.reason} during login.")
    logger.info("connected to CVAT at %s as %s", s.url, user.username)
    return client


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[Client]:
    client = _connect(settings)
    try:
        yield client
    finally:
        client.close()


WRITE_MODE = settings.mcp_mode == "write"
READ_ONLY_NOTE = (
    ""
    if WRITE_MODE
    else "Creating issues is disabled: the server runs in read-only mode "
    "(CVAT_MCP_MODE=write enables it)."
)

INSTRUCTIONS = f"""\
Tools for reviewing annotations in CVAT (computer vision annotation tool).
Mode: {settings.mcp_mode}.
Typical flow: list_tasks -> get_task_overview (labels, jobs and their frame ranges)
-> render_frame for each frame to review (look for missing objects, wrong labels and boxes
that miss their object) -> get_frame_annotations for exact ids and boxes
-> {"create_issue, one per problem" if WRITE_MODE else "report the problems to the user"}.
Issues belong to the job of type "annotation" whose frame range contains the frame.
All frame numbers are task frame numbers (0-based), not job-relative; coordinates are original
image pixels [x1, y1, x2, y2].
{READ_ONLY_NOTE}
Deleting or changing tasks, jobs and annotations is not available through this server in any
mode: if the user asks for it, explain that it is disabled here and must be done in the CVAT UI.
"""

mcp = MCPServer("cvat", instructions=INSTRUCTIONS, lifespan=lifespan)

REGISTERED_TOOLS: list[str] = []


def tool(
    *, write: bool = False, structured_output: bool | None = None, **hints
) -> Callable[[F], F]:
    """Register a tool, wrapped with model-facing error messages.

    Write tools are registered only in write mode: what isn't in the mode doesn't exist.
    """

    def decorator(fn: F) -> F:
        if write and not WRITE_MODE:
            return fn
        defaults = {"read_only_hint": not write, "open_world_hint": False}
        if write:
            defaults |= {"destructive_hint": False, "idempotent_hint": False}
        annotations = ToolAnnotations(**(defaults | hints))
        mcp.tool(annotations=annotations, structured_output=structured_output)(tool_errors(fn))
        REGISTERED_TOOLS.append(fn.__name__)
        return fn

    return decorator


def get_client(ctx: Context) -> Client:
    return ctx.request_context.lifespan_context
