from cvat_sdk import Client
from cvat_sdk.core.exceptions import IncompatibleVersionException
from cvat_sdk.version import VERSION as SDK_VERSION
from pydantic import BaseModel


class ServerInfo(BaseModel):
    server_version: str
    sdk_version: str
    compatible: bool
    compatibility_note: str | None
    user: str
    mode: str
    tools: list[str]


def check_compatibility(client: Client) -> tuple[bool, str | None]:
    # The SDK only logs or raises; there is no public "is compatible" boolean.
    try:
        client.check_server_version(fail_if_unsupported=True)
    except IncompatibleVersionException as e:
        return False, str(e)
    return True, None


def get_server_info(client: Client, mode: str, tools: list[str]) -> ServerInfo:
    compatible, note = check_compatibility(client)
    return ServerInfo(
        server_version=str(client.get_server_version()),
        sdk_version=SDK_VERSION,
        compatible=compatible,
        compatibility_note=note,
        user=client.users.retrieve_current_user().username,
        mode=mode,
        tools=tools,
    )
