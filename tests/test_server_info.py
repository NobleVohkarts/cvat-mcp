from cvat_sdk.core.exceptions import IncompatibleVersionException

from cvat_mcp.services.server_info import check_compatibility


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
