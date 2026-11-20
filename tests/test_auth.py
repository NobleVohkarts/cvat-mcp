"""OAuth provider: only the PAT owner gets tokens; codes and login requests are single-use."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as Obj
from urllib.parse import parse_qs, urlsplit

import anyio
import pytest
from cvat_sdk.api_client.exceptions import ApiValueError
from mcp.server.auth.provider import AuthorizationParams, RegistrationError
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient
from urllib3.exceptions import HTTPError

from cvat_mcp import auth
from cvat_mcp.auth import (
    MAX_CLIENTS,
    MAX_FAILURES,
    MAX_PENDING_PER_CLIENT,
    REFRESH_TTL,
    CvatOAuthProvider,
)

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
    csp = page.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    # browsers apply form-action to the redirect after the POST: the client's callback must pass
    assert "form-action 'self' https://claude.ai;" in csp
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


def test_registered_clients_are_capped_and_stripped(provider):
    info = OAuthClientInformationFull(
        client_id="first", redirect_uris=[CALLBACK], contacts=["x" * 1000], jwks={"k": "v" * 1000}
    )
    anyio.run(provider.register_client, info)
    stored = anyio.run(provider.get_client, "first")
    assert stored.contacts is None and stored.jwks is None
    for i in range(MAX_CLIENTS):
        anyio.run(
            provider.register_client,
            OAuthClientInformationFull(client_id=f"c{i}", redirect_uris=[CALLBACK]),
        )
    assert anyio.run(provider.get_client, "first") is None  # oldest evicted
    assert anyio.run(provider.get_client, f"c{MAX_CLIENTS - 1}") is not None


def test_pending_logins_are_purged_and_capped_per_client(provider, clock):
    start_login(provider)
    clock.t += 601
    for _ in range(MAX_PENDING_PER_CLIENT + 3):
        start_login(provider)
    assert len(provider._pending) == MAX_PENDING_PER_CLIENT  # pylint: disable=protected-access


def test_clients_holding_tokens_survive_a_registration_flood(provider, web):
    client, tokens = issue_tokens(provider, web)
    for i in range(MAX_CLIENTS + 10):
        anyio.run(
            provider.register_client,
            OAuthClientInformationFull(client_id=f"flood{i}", redirect_uris=[CALLBACK]),
        )
    assert anyio.run(provider.get_client, client.client_id) is not None
    assert anyio.run(provider.load_refresh_token, client, tokens.refresh_token) is not None


def test_too_many_redirect_uris_refused(provider):
    with pytest.raises(RegistrationError):
        anyio.run(
            provider.register_client,
            OAuthClientInformationFull(client_id="many", redirect_uris=[CALLBACK] * 6),
        )


def test_lockout_is_per_client_ip(provider, web):
    for _ in range(MAX_FAILURES):
        _, request_id, _ = start_login(provider)
        r = web.post(
            "/login",
            data={"request": request_id, "username": "admin", "password": "bad"},
            headers={"cf-connecting-ip": "203.0.113.7"},
        )
        assert r.status_code == 401
    _, request_id, _ = start_login(provider)
    attacker = web.post(
        "/login",
        data={"request": request_id, "username": "admin", "password": PASSWORD},
        headers={"cf-connecting-ip": "203.0.113.7"},
    )
    assert attacker.status_code == 429
    _, request_id, _ = start_login(provider)
    owner = web.post(
        "/login",
        data={"request": request_id, "username": "admin", "password": PASSWORD},
        headers={"cf-connecting-ip": "198.51.100.1"},
    )
    assert owner.status_code == 302


def test_old_token_pairs_are_purged(provider, web, clock):
    issue_tokens(provider, web)
    clock.t += REFRESH_TTL + 1
    issue_tokens(provider, web)
    assert len(provider._pairs) == 2  # pylint: disable=protected-access


def test_cvat_outage_during_login_shows_retry_page(clock):
    def down(_user, _password):
        raise HTTPError("cvat down")

    p = CvatOAuthProvider("https://x.trycloudflare.com", {"claude.ai"}, down, now=clock)
    p.owner = "admin"
    app = Starlette(routes=[Route("/login", p.login_submit, methods=["POST"])])
    _, request_id, _ = start_login(p)
    response = login(TestClient(app), request_id)
    assert response.status_code == 503 and "try again" in response.text


def test_cvat_login_treats_malformed_credentials_as_invalid(monkeypatch):
    class FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def login(self, _credentials):
            raise ApiValueError("password too short")

        def close(self):
            pass

    monkeypatch.setattr(auth, "Client", FakeClient)
    assert auth.cvat_login_username("http://cvat", "admin", "") is None


def test_logins_are_checked_one_at_a_time(clock, monkeypatch):
    monkeypatch.setattr(auth, "MAX_PENDING_PER_CLIENT", 20)  # 12 sign-in pages for one client
    active, peak, lock = [0], [0], threading.Lock()

    def slow_check(_user, _password):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1

    p = CvatOAuthProvider(
        "https://x.trycloudflare.com", {"claude.ai"}, slow_check, now=clock, failure_delay=0
    )
    p.owner = "admin"
    app = Starlette(routes=[Route("/login", p.login_submit, methods=["POST"])])
    ids = [start_login(p)[1] for _ in range(12)]
    with TestClient(app) as web, ThreadPoolExecutor(12) as pool:
        codes = list(pool.map(lambda rid: login(web, rid, password="bad").status_code, ids))
    assert peak[0] == 1
    assert codes.count(401) == MAX_FAILURES and codes.count(429) == 2


def test_login_body_is_limited(provider, web):
    _, request_id, _ = start_login(provider)
    response = web.post(
        "/login", data={"request": request_id, "username": "a" * 20_000, "password": "x"}
    )
    assert response.status_code == 413


def test_cvat_login_logs_out_and_skips_version_check(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, url, check_server_version=True):
            calls.append(("init", url, check_server_version))
            self.users = Obj(retrieve_current_user=lambda: Obj(username="admin"))

        def login(self, credentials):
            calls.append(("login", credentials[0]))

        def logout(self):
            calls.append("logout")

        def close(self):
            calls.append("close")

    monkeypatch.setattr(auth, "Client", FakeClient)
    assert auth.cvat_login_username("http://cvat", "admin", PASSWORD) == "admin"
    assert calls == [("init", "http://cvat", False), ("login", "admin"), "logout", "close"]
