"""OAuth for remote clients (claude.ai): the server is its own authorization server.

Only the CVAT user who owns the server's access token can log in, with their CVAT password,
checked once against CVAT and never stored. Everything lives in memory: a restart means the
client connects again.
"""

import html
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import anyio
from cvat_sdk import make_client
from cvat_sdk.api_client.exceptions import ApiException
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    RegistrationError,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response

SCOPE = "cvat"
LOGIN_TTL = 600
CODE_TTL = 300
ACCESS_TTL = 3600
REFRESH_TTL = 30 * 24 * 3600
MAX_FAILURES = 10
LOCKOUT = 600
LOCAL_HOSTS = {"localhost", "127.0.0.1"}

HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "Cache-Control": "no-store",
}


def cvat_login_username(cvat_url: str, username: str, password: str) -> str | None:
    """The CVAT username these credentials belong to, or None if CVAT rejects them."""
    try:
        with make_client(cvat_url, credentials=(username, password)) as client:
            return client.users.retrieve_current_user().username
    except ApiException as e:
        if e.status in (400, 401, 403):
            return None
        raise


@dataclass
class _PendingLogin:
    client_id: str
    params: AuthorizationParams
    expires_at: float


@dataclass
class _Failures:
    times: list[float] = field(default_factory=list)


class CvatOAuthProvider:  # pylint: disable=too-many-instance-attributes
    """mcp's OAuthAuthorizationServerProvider, plus the /login page."""

    def __init__(
        self,
        public_url: str,
        redirect_hosts: set[str],
        check_login: Callable[[str, str], str | None],
        now: Callable[[], float] = time.time,
        failure_delay: float = 1.0,
    ):
        self.public_url = public_url.rstrip("/")
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
        self._pairs: dict[str, str] = {}  # access token <-> refresh token, both directions
        self._failures = _Failures()

    # --- clients -------------------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        for uri in client_info.redirect_uris or []:
            parts = urlsplit(str(uri))
            local = parts.hostname in LOCAL_HOSTS
            if parts.hostname not in self.redirect_hosts or (parts.scheme != "https" and not local):
                raise RegistrationError(
                    error="invalid_redirect_uri",
                    error_description=f"redirect URI not allowed: {uri}",
                )
        self._clients[client_info.client_id] = client_info

    # --- authorization ---------------------------------------------------------------------

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        request_id = secrets.token_urlsafe(32)
        self._pending[request_id] = _PendingLogin(client.client_id, params, self.now() + LOGIN_TTL)
        return f"{self.public_url}/login?request={request_id}"

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._codes.get(authorization_code)
        if code is None or code.client_id != client.client_id or code.expires_at < self.now():
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        self._codes.pop(authorization_code.code, None)  # single use
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.subject)

    # --- tokens ------------------------------------------------------------------------------

    def _issue(self, client_id: str, scopes: list[str], subject: str | None) -> OAuthToken:
        now = int(self.now())
        access, refresh = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        self._access[access] = AccessToken(
            token=access,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + ACCESS_TTL,
            subject=subject,
        )
        self._refresh[refresh] = RefreshToken(
            token=refresh,
            client_id=client_id,
            scopes=scopes,
            expires_at=now + REFRESH_TTL,
            subject=subject,
        )
        self._pairs[access], self._pairs[refresh] = refresh, access
        return OAuthToken(
            access_token=access,
            expires_in=ACCESS_TTL,
            scope=" ".join(scopes),
            refresh_token=refresh,
        )

    async def load_access_token(self, token: str) -> AccessToken | None:
        access = self._access.get(token)
        if access is None or (access.expires_at or 0) < self.now():
            return None
        return access

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        refresh = self._refresh.get(refresh_token)
        if refresh is None or refresh.client_id != client.client_id:
            return None
        if (refresh.expires_at or 0) < self.now():
            return None
        return refresh

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str]
    ) -> OAuthToken:
        self._forget(refresh_token.token)  # rotate: the old pair stops working
        return self._issue(client.client_id, scopes or refresh_token.scopes, refresh_token.subject)

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        self._forget(token.token)

    async def exchange_identity_assertion(self, client, params) -> OAuthToken:
        raise TokenError(
            error="unsupported_grant_type", error_description="identity assertions not supported"
        )

    def _forget(self, token: str) -> None:
        other = self._pairs.pop(token, None)
        for value in (token, other):
            if value is not None:
                self._access.pop(value, None)
                self._refresh.pop(value, None)
                self._pairs.pop(value, None)

    # --- login page --------------------------------------------------------------------------

    def _pending_login(self, request_id: str) -> _PendingLogin | None:
        pending = self._pending.get(request_id)
        if pending is None or pending.expires_at < self.now():
            return None
        return pending

    def _locked_out(self) -> bool:
        recent = [t for t in self._failures.times if t > self.now() - LOCKOUT]
        self._failures.times = recent
        return len(recent) >= MAX_FAILURES

    async def login_form(self, request: Request) -> Response:
        request_id = request.query_params.get("request", "")
        if self._pending_login(request_id) is None:
            return _page("This login link is invalid or expired. Connect again from Claude.", 400)
        return _page(None, 200, request_id)

    async def login_submit(self, request: Request) -> Response:
        form = await request.form()
        request_id = str(form.get("request", ""))
        pending = self._pending_login(request_id)
        if pending is None:
            return _page("This login link is invalid or expired. Connect again from Claude.", 400)
        if self._locked_out():
            return _page("Too many failed logins. Try again in a few minutes.", 429, request_id)
        username, password = str(form.get("username", "")), str(form.get("password", ""))
        user = await anyio.to_thread.run_sync(self.check_login, username, password)
        if user is None or self.owner is None or user != self.owner:
            self._failures.times.append(self.now())
            await anyio.sleep(self.failure_delay)
            return _page("Invalid username or password.", 401, request_id)

        del self._pending[request_id]  # single use
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


def _page(message: str | None, status: int, request_id: str | None = None) -> HTMLResponse:
    form = ""
    if request_id is not None:
        form = f"""
<form method="post" action="login">
  <input type="hidden" name="request" value="{html.escape(request_id)}">
  <label>CVAT username <input name="username" autocomplete="username" required></label>
  <label>Password <input name="password" type="password" autocomplete="current-password"
    required></label>
  <button type="submit">Sign in</button>
</form>"""
    notice = f'<p class="error">{html.escape(message)}</p>' if message else ""
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>cvat-mcp sign in</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font: 16px system-ui, sans-serif; max-width: 22rem; margin: 4rem auto; padding: 0 1rem; }}
label {{ display: block; margin: .8rem 0; }} input {{ display: block; width: 100%; }}
.error {{ color: #b00020; }}
</style></head><body>
<h1>Sign in to cvat-mcp</h1>
<p>Claude wants to use your CVAT server. Sign in with the CVAT account that owns this
server's access token.</p>{notice}{form}
</body></html>"""
    return HTMLResponse(body, status_code=status, headers=HEADERS)
