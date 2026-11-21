from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.exceptions import IncompatibleVersionException

from cvat_mcp.services.server_info import check_compatibility, get_server_info


class FakeClient:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def check_server_version(self, fail_if_unsupported=None):
        self.calls.append(fail_if_unsupported)
        if self.error:
            raise self.error


def test_compatible_server():
    client = FakeClient()
    assert check_compatibility(client) == (True, None)
    assert client.calls == [True]


def test_incompatible_server_returns_sdk_message():
    message = "Server version '2.70.0' is not compatible with SDK version '2.83.0'."
    assert check_compatibility(FakeClient(IncompatibleVersionException(message))) == (
        False,
        message,
    )


def test_server_info_survives_unreachable_about_endpoint():
    class Users:
        def retrieve_current_user(self):
            return type("U", (), {"username": "admin"})()

    class Broken(FakeClient):
        users = Users()

        def get_server_version(self):
            raise ApiException(status=502, reason="Bad Gateway")

    client = Broken(IncompatibleVersionException("Failed to retrieve server API version"))
    info = get_server_info(client, "read-only", ["server_info"])
    assert info.server_version == "unknown"
    assert info.compatible is False


def test_server_info_reports_the_cvat_mcp_version():
    import cvat_mcp  # pylint: disable=import-outside-toplevel

    class Users:
        def retrieve_current_user(self):
            return type("U", (), {"username": "admin"})()

    class Ok(FakeClient):
        users = Users()

        def get_server_version(self):
            return "2.83.0"

    info = get_server_info(Ok(), "read-only", ["server_info"])
    assert info.cvat_mcp_version == cvat_mcp.__version__
