import functools
import logging
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.util import find_spec
from typing import TypeVar
from urllib.parse import urlsplit

from cvat_sdk import Client, make_client
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, ToolAnnotations
from urllib3.exceptions import HTTPError

from cvat_mcp import __version__
from cvat_mcp.auth import SCOPE, CvatOAuthProvider, cvat_login_username
from cvat_mcp.config import Settings
from cvat_mcp.errors import explain
from cvat_mcp.tasks import Tasks

logger = logging.getLogger("cvat_mcp")

settings = Settings()


def _connect(s: Settings) -> tuple[Client, str]:
    # make_client() also checks the server version and logs a warning if it is unsupported.
    try:
        client = open_client(s)
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
    return client, user.username


def open_client(s: Settings = settings) -> Client:
    """A new CVAT client (also for background runs, which outlive a request); raises, never exits."""
    return make_client(s.url, access_token=s.access_token.get_secret_value())


def check_cvat() -> None:
    """Exit with one line if CVAT is unreachable or rejects the token."""
    client, _ = _connect(settings)
    client.close()


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[Client]:
    client, username = _connect(settings)
    if oauth is not None:
        oauth.owner = username  # only the token's own CVAT user may log in through OAuth
        logger.info("serving %s/mcp (OAuth login: CVAT user %s)", settings.public_url, username)
    try:
        yield client
    finally:
        client.close()


WRITE_MODE = settings.mcp_mode == "write"
# The auto-annotation tools need the optional extra: `uv sync --extra auto-annotation`.
AUTO_ANNOTATION = all(find_spec(m) is not None for m in ("torch", "torchvision", "skimage"))
AUTO_ANNOTATION_STEPS = (
    "; auto-annotate (auto_annotate) only when the user asks: with a frame it waits and returns "
    "the objects (check them with render_frame), without one it runs over the whole task in "
    "the background (follow it with get_auto_annotation_run)"
    if AUTO_ANNOTATION
    else ""
)
if WRITE_MODE:
    WRITE_STEPS = (
        " -> create_issue, one per problem, only when the user asks to file issues "
        '(reviewing is not such a request). Issues belong to the job of type "annotation" '
        "whose frame range contains the frame. To rewrite or replace issues, list_issues for the "
        "frame, create_issue the new ones, then delete_issue the old ones"
        ". To answer, resolve or reopen issues (update_issue), list_issues first; resolve issues "
        "only when the user asks"
        ". Fix annotations with create_annotations / update_annotations / delete_annotations "
        "only when the user asks to fix them (reviewing or filing issues is not such a request), "
        "then render_frame again to check. Create, change or delete tasks (create_task, "
        "update_task, delete_task) only when the user asks; delete_task removes only tasks owned "
        "by this server's CVAT account. Change jobs (update_job) or create and delete "
        "ground-truth jobs (create_gt_job, delete_gt_job) only when the user asks; delete_gt_job "
        "removes only ground-truth jobs of tasks owned by this server's CVAT account. "
        "Ground-truth annotations: pass job_id of the ground-truth job to the annotation tools; "
        "edit ground truth only when the user explicitly asks"
        "; switch quality requirements on (update_quality_settings) only when the user asks or "
        "create_report says nothing is enabled and the user wants the report"
        "; register, change or delete cloud storages (create_cloud_storage, "
        "update_cloud_storage, delete_cloud_storage) only when the user asks; keys go only to "
        "create_cloud_storage and update_cloud_storage, never repeat keys back"
        "; to use a bucket (create_task, import_annotations, export_dataset), "
        "list_cloud_storages for the "
        "bucket's id and pass it with cloud_path; register a missing bucket only when the user "
        "asks"
        "; import annotation files (import_annotations) only when the user asks; it appends "
        "unless the user asks to replace"
        "; export datasets (export_dataset) into a bucket only when the user asks; it refuses "
        "an existing cloud_path: ask the user before overwrite=true" + AUTO_ANNOTATION_STEPS
    )
    READ_ONLY_NOTE = ""
else:
    WRITE_STEPS = ""
    READ_ONLY_NOTE = (
        "Write tools are disabled: the server runs in read-only mode (CVAT_MCP_MODE=write "
        "enables them)."
    )

INSTRUCTIONS = f"""\
Tools for reviewing annotations in CVAT (computer vision annotation tool) and, in write mode,
for fixing them and managing tasks and jobs.
Mode: {settings.mcp_mode}.
Typical flow: list_tasks (or list_jobs, e.g. stage "validation", for the jobs waiting for review)
-> get_task_overview (labels, jobs and their frame ranges)
-> render_frame for each frame to review (look for missing objects, wrong labels and boxes
that miss their object) -> get_frame_annotations for exact ids and boxes
(download_annotations for all of a job's annotations as JSON)
-> list_issues for issues already filed there -> report the problems to the user{WRITE_STEPS}.
To measure annotation jobs against the task's ground-truth job, call create_report, then
render_frame the conflict frames, once as they are and once with
job_id of the ground-truth job to see the ground truth.
All frame numbers are task frame numbers (0-based), not job-relative; coordinates are original
image pixels [x1, y1, x2, y2].
{READ_ONLY_NOTE}
"""

HTTP_MODE = settings.mcp_transport == "http"
oauth: CvatOAuthProvider | None = None
auth: AuthSettings | None = None
if HTTP_MODE:
    oauth = CvatOAuthProvider(
        settings.public_url,
        settings.redirect_hosts,
        functools.partial(cvat_login_username, settings.url),
    )
    auth = AuthSettings(
        issuer_url=settings.public_url,
        resource_server_url=f"{settings.public_url}/mcp",
        client_registration_options=ClientRegistrationOptions(
            enabled=True, valid_scopes=[SCOPE], default_scopes=[SCOPE]
        ),
        revocation_options=RevocationOptions(enabled=True),
        required_scopes=[SCOPE],
        validate_token_resource=False,  # this issuer only mints tokens for this one server
    )


async def reject_unknown_arguments(ctx, call_next):
    """mcp drops unknown tool arguments silently; a client with a stale tool list would then
    never learn that its call didn't do what it asked. Refuse them with the valid names.

    Raised as JSON-RPC invalid params (what the MCP spec calls invalid arguments): mcp's runner
    shapes errors for every protocol version, which a result built here would bypass.
    """
    params = ctx.params if isinstance(ctx.params, dict) else {}
    arguments = params.get("arguments")
    if ctx.method == "tools/call" and isinstance(arguments, dict):  # else mcp validates
        name = params.get("name")
        # list_tools() is public; mcp's by-name lookup is private, and this costs microseconds
        tool = next((t for t in await mcp.list_tools() if t.name == name), None)
        valid = set(tool.input_schema.get("properties", {})) if tool else set()
        unknown = sorted(set(arguments) - valid)
        if tool and unknown:
            takes = f"valid: {', '.join(sorted(valid))}" if valid else "it takes no arguments"
            raise MCPError(
                code=INVALID_PARAMS,
                message=(
                    f"unknown argument(s) {', '.join(unknown)} for {name}; {takes}. "
                    "Your tool list may be out of date: reconnect the connector to refresh it."
                ),
            )
    return await call_next(ctx)


mcp = MCPServer(
    "cvat",
    version=__version__,
    instructions=INSTRUCTIONS,
    lifespan=lifespan,
    auth_server_provider=oauth,
    auth=auth,
    middleware=[reject_unknown_arguments],
    extensions=[Tasks()],
)
if oauth is not None:
    mcp.custom_route("/login", ["GET"])(oauth.login_form)
    mcp.custom_route("/login", ["POST"])(oauth.login_submit)


def transport_security() -> TransportSecuritySettings:
    """Accept only the public (tunnel or proxy) host and the local address as Host headers."""
    port = settings.mcp_port
    hosts = [f"127.0.0.1:{port}", f"localhost:{port}", urlsplit(settings.public_url).netloc]
    # browser-based clients (e.g. MCP Inspector) send an Origin header
    origins = [settings.public_url, "http://localhost:*", "http://127.0.0.1:*"]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins
    )


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
    *, write: bool = False, enabled: bool = True, structured_output: bool | None = None, **hints
) -> Callable[[F], F]:
    """Register a tool, wrapped with model-facing error messages.

    Write tools are registered only in write mode, and `enabled=False` tools (a missing optional
    extra) not at all: what isn't available doesn't exist.
    """

    def decorator(fn: F) -> F:
        if not enabled or (write and not WRITE_MODE):
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
