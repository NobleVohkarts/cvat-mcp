from mcp.server.mcpserver import Context

from cvat_mcp.server import REGISTERED_TOOLS, get_client, settings, tool
from cvat_mcp.services.server_info import ServerInfo, get_server_info


@tool()
def server_info(ctx: Context) -> ServerInfo:
    """Use first, or when something fails: shows the CVAT server and SDK versions, whether they
    are compatible, who you are logged in as, the server mode and the available tools."""
    return get_server_info(get_client(ctx), settings.mcp_mode, REGISTERED_TOOLS)
