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


INSTRUCTIONS = f"""\
Tools for reviewing annotations in CVAT (computer vision annotation tool).
Mode: {settings.mcp_mode}.
Typical flow: list_tasks -> get_task_overview (labels, jobs and their frame ranges).
All frame numbers are task frame numbers (0-based), not job-relative.
Deleting or changing tasks, jobs and annotations is not available through this server:
if the user asks for it, explain that it is disabled here and must be done in the CVAT UI.
"""

mcp = MCPServer("cvat", instructions=INSTRUCTIONS, lifespan=lifespan)

REGISTERED_TOOLS: list[str] = []


def tool[F: Callable](**hints) -> Callable[[F], F]:
    """Register a read-only tool with MCP annotations. Hints override the read-only defaults."""

    def decorator(fn: F) -> F:
        annotations = ToolAnnotations(**{"read_only_hint": True, "open_world_hint": False, **hints})
        mcp.tool(annotations=annotations)(fn)
        REGISTERED_TOOLS.append(fn.__name__)
        return fn

    return decorator


def get_client(ctx: Context) -> Client:
    return ctx.request_context.lifespan_context
