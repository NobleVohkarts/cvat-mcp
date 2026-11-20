import functools
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TypeVar
from urllib.parse import urlsplit

from cvat_sdk import Client, make_client
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from urllib3.exceptions import HTTPError

from cvat_mcp.auth import SCOPE, CvatOAuthProvider, cvat_login_username
from cvat_mcp.config import Settings
from cvat_mcp.errors import explain

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
    if oauth is not None:
        oauth.owner = user.username  # only this CVAT user may log in through OAuth
        logger.info("serving %s/mcp (OAuth login: CVAT user %s)", s.public_url, user.username)
    return client


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[Client]:
    client = _connect(settings)
    try:
        yield client
    finally:
        client.close()


WRITE_MODE = settings.mcp_mode == "write"
if WRITE_MODE:
    FILE_ISSUES = (
        " -> create_issue, one per problem, only when the user asks to file issues "
        '(reviewing is not such a request). Issues belong to the job of type "annotation" '
        "whose frame range contains the frame"
    )
    READ_ONLY_NOTE = ""
else:
    FILE_ISSUES = ""
    READ_ONLY_NOTE = (
        "Creating issues is disabled: the server runs in read-only mode "
        "(CVAT_MCP_MODE=write enables it)."
    )

INSTRUCTIONS = f"""\
Tools for reviewing annotations in CVAT (computer vision annotation tool).
Mode: {settings.mcp_mode}.
Typical flow: list_tasks -> get_task_overview (labels, jobs and their frame ranges)
-> render_frame for each frame to review (look for missing objects, wrong labels and boxes
that miss their object) -> get_frame_annotations for exact ids and boxes
-> report the problems to the user{FILE_ISSUES}.
All frame numbers are task frame numbers (0-based), not job-relative; coordinates are original
image pixels [x1, y1, x2, y2].
{READ_ONLY_NOTE}
Deleting or changing tasks, jobs and annotations is not available through this server in any
mode: if the user asks for it, explain that it is disabled here and must be done in the CVAT UI.
"""

HTTP_MODE = settings.mcp_transport == "http"
oauth = (
    CvatOAuthProvider(
        settings.public_url,
        settings.redirect_hosts,
        functools.partial(cvat_login_username, settings.url),
    )
    if HTTP_MODE
    else None
)
auth = (
    AuthSettings(
        issuer_url=settings.public_url,
        resource_server_url=f"{settings.public_url}/mcp",
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
        ),
        revocation_options=RevocationOptions(enabled=True),
        required_scopes=[SCOPE],
        validate_token_resource=False,  # this issuer only mints tokens for this one server
    )
    if HTTP_MODE
    else None
)
mcp = MCPServer(
    "cvat", instructions=INSTRUCTIONS, lifespan=lifespan, auth_server_provider=oauth, auth=auth
)
if oauth is not None:
    mcp.custom_route("/login", ["GET"])(oauth.login_form)
    mcp.custom_route("/login", ["POST"])(oauth.login_submit)


def transport_security() -> TransportSecuritySettings:
    """Accept only the public (tunnel) host and the local address as Host headers."""
    port = settings.mcp_port
    hosts = [f"127.0.0.1:{port}", f"localhost:{port}"]
    if settings.public_url:
        hosts.append(urlsplit(settings.public_url).netloc)
    return TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts)


REGISTERED_TOOLS: list[str] = []
F = TypeVar("F", bound=Callable)


def tool_errors(fn: F) -> F:
    """Re-raise expected failures as ToolError so their message reaches the model."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            message = explain(exc)
            if message is None:
                raise
            raise ToolError(message) from exc

    return wrapper


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
