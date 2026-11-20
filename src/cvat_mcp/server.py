import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TypeVar

from cvat_sdk import Client, make_client
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations
from pydantic import ValidationError
from urllib3.exceptions import HTTPError

from cvat_mcp.config import Settings
from cvat_mcp.services.server_info import check_compatibility

logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # stdout belongs to the MCP protocol
logging.getLogger("cvat_sdk").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.ERROR)  # retry warnings repeat our one-line error
logger = logging.getLogger("cvat_mcp")


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as e:
        fields = ", ".join(f"CVAT_{str(err['loc'][0]).upper()}" for err in e.errors())
        sys.exit(
            f"cvat-mcp: invalid or missing {fields}. Create a Personal Access Token in the CVAT UI "
            "and put CVAT_ACCESS_TOKEN=... into cvat-mcp/.env (see .env.example)."
        )


settings = _load_settings()


@dataclass
class AppContext:
    client: Client
    settings: Settings


def _connect(s: Settings) -> Client:
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
    compatible, note = check_compatibility(client)
    if not compatible:
        logger.warning(note)
    logger.info("connected to CVAT %s at %s as %s", client.get_server_version(), s.url, user.username)
    return client


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
    client = _connect(settings)
    try:
        yield AppContext(client=client, settings=settings)
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

F = TypeVar("F", bound=Callable)


def tool(**hints) -> Callable[[F], F]:
    """Register a read-only tool with MCP annotations. Hints override the read-only defaults."""

    def decorator(fn: F) -> F:
        annotations = ToolAnnotations(
            **{"read_only_hint": True, "open_world_hint": False, **hints}
        )
        mcp.tool(annotations=annotations)(fn)
        REGISTERED_TOOLS.append(fn.__name__)
        return fn

    return decorator


def get_client(ctx: Context) -> Client:
    return ctx.request_context.lifespan_context.client
