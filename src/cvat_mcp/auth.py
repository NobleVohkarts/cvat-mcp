"""OAuth for remote clients (claude.ai): the server is its own authorization server.

Only the CVAT user who owns the server's access token can log in, with their CVAT password,
checked once against CVAT and never stored. Everything lives in memory, bounded so the public
endpoint can't be used to fill it; a restart means the client connects again.
"""

import functools
import html
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs

import anyio
from cvat_sdk.api_client.exceptions import ApiException
from cvat_sdk.core.client import Client
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    RegistrationError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

from cvat_mcp.config import is_secure_url

SCOPE = "cvat"
LOGIN_TTL = 600
CODE_TTL = 300
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 24 * 3600
MAX_FAILURES = 10
LOCKOUT = 600
MAX_CLIENTS = 100  # open dynamic registration: keep the newest, drop the oldest
MAX_PENDING = 50
MAX_FORM_BYTES = 16 * 1024
# Free-form client metadata we never use; not kept, so registrations stay small.
UNUSED_CLIENT_FIELDS = (
    "contacts", "jwks", "jwks_uri", "logo_uri", "policy_uri", "tos_uri", "client_uri",
    "software_id", "software_version",
)  # fmt: skip
EXPIRED = "This login link is invalid or expired. Connect again from Claude."

HEADERS = {"X-Frame-Options": "DENY", "Cache-Control": "no-store"}


def cvat_login_username(cvat_url: str, username: str, password: str) -> str | None:
    """The CVAT username these credentials belong to, or None if CVAT rejects them."""
    client = Client(cvat_url, check_server_version=False)  # checked once at startup
    try:
        client.login((username, password))
        try:
            return client.users.retrieve_current_user().username
        finally:
            client.logout()  # don't leave a CVAT session behind
    except ApiException as e:
        if e.status in (400, 401, 403):
            return None
        raise
    finally:
        client.close()


@dataclass
class _PendingLogin:
    client_id: str
    params: AuthorizationParams
    expires_at: float


class CvatOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):  # pylint: disable=too-many-instance-attributes
    """mcp's authorization server provider, plus the /login page."""

    def __init__(
        self,
        public_url: str,
        redirect_hosts: set[str],
        check_login: Callable[[str, str], str | None],
        now: Callable[[], float] = time.time,
        failure_delay: float = 1.0,
    ):
        self.public_url = public_url
        self.redirect_hosts = redirect_hosts
        self.check_login = check_login
        self.now = now
        self.failure_delay = failure_delay
        self.owner: str | None = None  # the PAT's CVAT user; set by the server's lifespan
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._pending: dict[str, _PendingLogin] = {}
        self._codes: dict[str, AuthorizationCode] = {}
        self._access: dict[str, AccessToken] = {}
        self._refresh: dict[str, RefreshToken] = {}
        self._pairs: dict[str, tuple[str, str]] = {}  # either token -> (access, refresh)
        self._failures: list[float] = []
        self._login_lock = anyio.Lock()  # one login at a time: exact lockout, no thread pile-up

    def _live(self, store: dict, key: str):
        """The item, or None if it's missing or expired (expired items are dropped)."""
        item = store.get(key)
        if item is not None and (item.expires_at or 0) < self.now():
            store.pop(key, None)
            return None
        return item

    def _purge(self, store: dict, limit: int | None = None) -> None:
        now = self.now()
        for key in [k for k, v in store.items() if (v.expires_at or 0) < now]:
            del store[key]
        while limit is not None and len(store) >= limit:
            store.pop(next(iter(store)))  # dicts keep insertion order: drop the oldest

    # --- clients -------------------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        for uri in client_info.redirect_uris or []:
            if not is_secure_url(str(uri), self.redirect_hosts):
                raise RegistrationError(
                    error="invalid_redirect_uri",
                    error_description=f"redirect URI not allowed: {uri}",
                )
        while len(self._clients) >= MAX_CLIENTS:
            self._clients.pop(next(iter(self._clients)))
        self._clients[client_info.client_id] = client_info.model_copy(
            update=dict.fromkeys(UNUSED_CLIENT_FIELDS)
        )

    # --- authorization ---------------------------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        self._purge(self._pending, MAX_PENDING)
        request_id = secrets.token_urlsafe(32)
        self._pending[request_id] = _PendingLogin(client.client_id, params, self.now() + LOGIN_TTL)
        return f"{self.public_url}/login?request={request_id}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._live(self._codes, authorization_code)
        return code if code is not None and code.client_id == client.client_id else None

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        self._codes.pop(authorization_code.code, None)  # single use
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.subject)

    # --- tokens ------------------------------------------------------------------------------

    def _issue(self, client_id: str, scopes: list[str], subject: str | None) -> OAuthToken:
        self._purge(self._access)
        self._purge(self._refresh)
        now = int(self.now())
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        common = {"client_id": client_id, "scopes": scopes, "subject": subject}
        self._access[access] = AccessToken(token=access, expires_at=now + ACCESS_TTL, **common)
        self._refresh[refresh] = RefreshToken(token=refresh, expires_at=now + REFRESH_TTL, **common)
        self._pairs[access] = self._pairs[refresh] = (access, refresh)
        return OAuthToken(
            access_token=access,
            expires_in=ACCESS_TTL,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        return self._live(self._access, token)

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        refresh = self._live(self._refresh, refresh_token)
        return refresh if refresh is not None and refresh.client_id == client.client_id else None

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        self._forget(refresh_token.token)  # rotate: the old pair stops working
        return self._issue(client.client_id, scopes or refresh_token.scopes, refresh_token.subject)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._forget(token.token)

    def _forget(self, token: str) -> None:
        for value in self._pairs.pop(token, ()):
            self._pairs.pop(value, None)
            self._access.pop(value, None)
            self._refresh.pop(value, None)

    # --- login page --------------------------------------------------------------------------

    async def login_form(self, request: Request) -> Response:
        request_id = request.query_params.get("request", "")
        pending = self._live(self._pending, request_id)
        if pending is None:
            return _page(EXPIRED, 400)
        return _page(None, 200, request_id, str(pending.params.redirect_uri))

    async def login_submit(self, request: Request) -> Response:
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_FORM_BYTES:
                return _page("The login form was too large.", 413)
        form = {k: v[0] for k, v in parse_qs(body.decode(errors="replace")).items()}
        request_id = form.get("request", "")
        pending = self._live(self._pending, request_id)
        if pending is None:
            return _page(EXPIRED, 400)
        retry = functools.partial(
            _page, request_id=request_id, redirect_uri=str(pending.params.redirect_uri)
        )

        async with self._login_lock:
            self._failures = [t for t in self._failures if t > self.now() - LOCKOUT]
            if len(self._failures) >= MAX_FAILURES:
                return retry("Too many failed logins. Try again in a few minutes.", 429)
            user = await anyio.to_thread.run_sync(
                self.check_login, form.get("username", ""), form.get("password", "")
            )
            if user is None or user != self.owner:
                self._failures.append(self.now())
                await anyio.sleep(self.failure_delay)
                return retry("Invalid username or password.", 401)

        if self._pending.pop(request_id, None) is None:  # single use, even across a race
            return _page(EXPIRED, 400)
        self._purge(self._codes)
        code = secrets.token_urlsafe(32)
        params = pending.params
        self._codes[code] = AuthorizationCode(
            code=code,
            scopes=params.scopes or [SCOPE],
            expires_at=self.now() + CODE_TTL,
            client_id=pending.client_id,
            code_challenge=params.code_challenge,
            redirect_uri=params.redirect_uri,
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            resource=params.resource,
            subject=user,
        )
        target = construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)
        return RedirectResponse(target, status_code=302, headers=HEADERS)


PAGE_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>cvat-mcp sign in</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { font: 16px system-ui, sans-serif; max-width: 22rem; margin: 4rem auto; padding: 0 1rem; }
label { display: block; margin: .8rem 0; } input { display: block; width: 100%; }
.error { color: #b00020; }
</style></head><body>
<h1>Sign in to cvat-mcp</h1>
<p>Claude wants to use your CVAT server. Sign in with the CVAT account that owns this
server's access token.</p>"""

PAGE_FORM = """
<form method="post" action="login">
  <input type="hidden" name="request" value="{request}">
  <label>CVAT username <input name="username" autocomplete="username" required></label>
  <label>Password <input name="password" type="password" autocomplete="current-password"
    required></label>
  <button type="submit">Sign in</button>
</form>"""


def _headers(redirect_uri: str | None) -> dict[str, str]:
    # Browsers apply form-action to the redirect that follows the POST, so the client's own
    # callback origin (already limited to the allowed redirect hosts) must be allowed too.
    form_action = "'self'"
    if redirect_uri:
        scheme, rest = redirect_uri.split("://", 1)
        form_action += f" {scheme}://{rest.split('/', 1)[0]}"
    csp = (
        "default-src 'none'; style-src 'unsafe-inline'; "
        f"form-action {form_action}; frame-ancestors 'none'"
    )
    return {**HEADERS, "Content-Security-Policy": csp}


def _page(
    message: str | None,
    status: int,
    request_id: str | None = None,
    redirect_uri: str | None = None,
) -> HTMLResponse:
    notice = f'<p class="error">{html.escape(message)}</p>' if message else ""
    form = PAGE_FORM.format(request=html.escape(request_id)) if request_id is not None else ""
    return HTMLResponse(
        f"{PAGE_HEAD}{notice}{form}\n</body></html>",
        status_code=status,
        headers=_headers(redirect_uri),
    )
