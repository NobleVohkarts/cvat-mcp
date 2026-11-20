"""OAuth provider: only the PAT owner gets tokens; codes and login requests are single-use."""

from urllib.parse import parse_qs, urlsplit

import anyio
import pytest
from mcp.server.auth.provider import AuthorizationParams, RegistrationError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from cvat_mcp.auth import CvatOAuthProvider

CALLBACK = "https://claude.ai/api/mcp/auth_callback"
PASSWORD = "s3cret-pass"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def check_login(username, password):
    """Fake CVAT login: admin and alice exist."""
    return username if password == PASSWORD and username in ("admin", "alice") else None


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def provider(clock):
    p = CvatOAuthProvider(
        "https://x.trycloudflare.com",
        {"claude.ai", "claude.com", "localhost", "127.0.0.1"},
        check_login,
        now=clock,
        failure_delay=0,
    )
    p.owner = "admin"
    return p


@pytest.fixture
def web(provider):
    app = Starlette(
        routes=[
            Route("/login", provider.login_form, methods=["GET"]),
            Route("/login", provider.login_submit, methods=["POST"]),
        ]
    )
    return TestClient(app, follow_redirects=False)


def client_info(redirect=CALLBACK):
    return OAuthClientInformationFull(client_id="c1", redirect_uris=[redirect])


def register(provider, redirect=CALLBACK):
    anyio.run(provider.register_client, client_info(redirect))
    return anyio.run(provider.get_client, "c1")


def start_login(provider):
    client = register(provider)
    params = AuthorizationParams(
        state="st",
        scopes=["cvat"],
        code_challenge="challenge",
        redirect_uri=AnyUrl(CALLBACK),
        redirect_uri_provided_explicitly=True,
    )
    url = anyio.run(provider.authorize, client, params)
    return client, parse_qs(urlsplit(url).query)["request"][0], url


def login(web, request_id, username="admin", password=PASSWORD):
    return web.post(
        "/login", data={"request": request_id, "username": username, "password": password}
    )


def code_from(response):
    return parse_qs(urlsplit(response.headers["location"]).query)["code"][0]


@pytest.mark.parametrize("redirect", ["https://evil.example/cb", "http://claude.ai/cb"])
def test_registration_refuses_foreign_or_insecure_redirects(provider, redirect):
    with pytest.raises(RegistrationError):
        anyio.run(provider.register_client, client_info(redirect))


def test_local_http_redirect_allowed_for_testing(provider):
    assert register(provider, "http://localhost:9/callback") is not None


def test_authorize_sends_user_to_login_page(provider):
    _, _, url = start_login(provider)
    assert url.startswith("https://x.trycloudflare.com/login?request=")


def test_owner_login_redirects_with_code_and_state(provider, web):
    _, request_id, _ = start_login(provider)
    response = login(web, request_id)
    assert response.status_code == 302
    query = parse_qs(urlsplit(response.headers["location"]).query)
    assert response.headers["location"].startswith(CALLBACK)
    assert query["state"] == ["st"] and query["code"]


@pytest.mark.parametrize(
    "username, password", [("admin", "wrong"), ("alice", PASSWORD), ("nobody", PASSWORD)]
)
def test_only_the_pat_owner_can_log_in(provider, web, username, password):
    _, request_id, _ = start_login(provider)
    response = login(web, request_id, username, password)
    assert response.status_code == 401
    assert "Invalid username or password" in response.text
    assert password not in response.text


def test_login_request_is_single_use(provider, web):
    _, request_id, _ = start_login(provider)
    assert login(web, request_id).status_code == 302
    assert login(web, request_id).status_code == 400


def test_login_request_expires(provider, web, clock):
    _, request_id, _ = start_login(provider)
    clock.t += 601
    assert web.get(f"/login?request={request_id}").status_code == 400
    assert login(web, request_id).status_code == 400


def test_login_page_headers_and_escaping(provider, web):
    _, request_id, _ = start_login(provider)
    page = web.get(f"/login?request={request_id}")
    assert page.status_code == 200
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    assert web.get('/login?request="><script>').status_code == 400


def test_code_is_single_use_and_expires(provider, web, clock):
    client, request_id, _ = start_login(provider)
    code = code_from(login(web, request_id))
    loaded = anyio.run(provider.load_authorization_code, client, code)
    assert loaded.code_challenge == "challenge"
    tokens = anyio.run(provider.exchange_authorization_code, client, loaded)
    assert tokens.access_token and tokens.refresh_token and tokens.scope == "cvat"
    assert anyio.run(provider.load_authorization_code, client, code) is None

    _, request_id, _ = start_login(provider)
    code = code_from(login(web, request_id))
    clock.t += 301
    assert anyio.run(provider.load_authorization_code, client, code) is None


def issue_tokens(provider, web):
    client, request_id, _ = start_login(provider)
    code = anyio.run(provider.load_authorization_code, client, code_from(login(web, request_id)))
    return client, anyio.run(provider.exchange_authorization_code, client, code)


def test_access_token_expires(provider, web, clock):
    _, tokens = issue_tokens(provider, web)
    assert anyio.run(provider.load_access_token, tokens.access_token).subject == "admin"
    clock.t += 3601
    assert anyio.run(provider.load_access_token, tokens.access_token) is None


def test_refresh_rotates_tokens(provider, web):
    client, tokens = issue_tokens(provider, web)
    old = anyio.run(provider.load_refresh_token, client, tokens.refresh_token)
    new = anyio.run(provider.exchange_refresh_token, client, old, ["cvat"])
    assert new.access_token != tokens.access_token
    assert anyio.run(provider.load_refresh_token, client, tokens.refresh_token) is None
    assert anyio.run(provider.load_access_token, tokens.access_token) is None


def test_revoke_removes_both_tokens(provider, web):
    client, tokens = issue_tokens(provider, web)
    access = anyio.run(provider.load_access_token, tokens.access_token)
    anyio.run(provider.revoke_token, access)
    assert anyio.run(provider.load_access_token, tokens.access_token) is None
    assert anyio.run(provider.load_refresh_token, client, tokens.refresh_token) is None


def test_lockout_after_ten_failures(provider, web, clock):
    for _ in range(10):
        _, request_id, _ = start_login(provider)
        assert login(web, request_id, password="wrong").status_code == 401
    _, request_id, _ = start_login(provider)
    assert login(web, request_id).status_code == 429
    clock.t += 601
    _, request_id, _ = start_login(provider)
    assert login(web, request_id).status_code == 302
