"""Smoke test for HTTP mode: the full OAuth flow a client like claude.ai runs, then tool calls.

Usage (from cvat-mcp/): uv run python scripts/check_remote.py <base-url>
Reads CVAT_UI_USERNAME / CVAT_UI_PASSWORD from .env (the server never reads them); prints no
secrets. Exits non-zero at the first failing step.
"""

import asyncio
import sys
from urllib.parse import parse_qs, urlsplit

import httpx2
from dotenv import dotenv_values
from mcp.client.auth.oauth2 import PKCEParameters
from mcp.client.auth.utils import extract_resource_metadata_from_www_auth
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata

CALLBACK = "http://localhost:9/callback"


def env(name: str) -> str:
    value = dotenv_values(".env").get(name)
    if not value:
        sys.exit(f"{name} missing in .env")
    return value


def step(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        sys.exit(1)


async def oauth_token(http: httpx2.AsyncClient, base: str) -> str:
    probe = await http.post(f"{base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    meta_url = extract_resource_metadata_from_www_auth(probe)
    step(probe.status_code == 401 and meta_url is not None, "/mcp without a token -> 401")
    resource = (await http.get(meta_url)).json()
    issuer = resource["authorization_servers"][0].rstrip("/")
    server = (await http.get(f"{issuer}/.well-known/oauth-authorization-server")).json()
    step("registration_endpoint" in server, f"discovered authorization server {issuer}")

    registered = await http.post(
        server["registration_endpoint"],
        json=OAuthClientMetadata(
            redirect_uris=[CALLBACK],
            token_endpoint_auth_method="none",
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            client_name="cvat-mcp smoke test",
        ).model_dump(mode="json", exclude_none=True),
    )
    step(registered.status_code == 201, "dynamic client registration")
    client_id = registered.json()["client_id"]

    pkce = PKCEParameters.generate()
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": CALLBACK,
        "code_challenge": pkce.code_challenge,
        "code_challenge_method": "S256",
        "state": "smoke",
        "scope": "cvat",
        "resource": f"{base}/mcp",
    }
    authorize = await http.get(server["authorization_endpoint"], params=params)
    login_url = authorize.headers.get("location", "")
    step(authorize.status_code == 302 and "/login?request=" in login_url, "authorize -> login")
    request_id = parse_qs(urlsplit(login_url).query)["request"][0]

    form = {"request": request_id, "username": env("CVAT_UI_USERNAME")}
    wrong = await http.post(login_url, data={**form, "password": "definitely-wrong"})
    step(wrong.status_code == 401, "wrong password refused")
    login = await http.post(login_url, data={**form, "password": env("CVAT_UI_PASSWORD")})
    location = login.headers.get("location", "")
    step(login.status_code == 302 and location.startswith(CALLBACK), "CVAT login -> callback")
    code = parse_qs(urlsplit(location).query)["code"][0]

    token = await http.post(
        server["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": CALLBACK,
            "client_id": client_id,
            "code_verifier": pkce.code_verifier,
        },
    )
    step(token.status_code == 200, "code exchanged for an access token")
    return token.json()["access_token"]


async def main(base: str) -> None:
    async with httpx2.AsyncClient(follow_redirects=False, timeout=60) as http:
        access = await oauth_token(http, base)
    headers = {"Authorization": f"Bearer {access}"}
    async with httpx2.AsyncClient(headers=headers, timeout=60) as http:
        async with streamable_http_client(f"{base}/mcp", http_client=http) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = sorted(t.name for t in (await session.list_tools()).tools)
                step(bool(tools), f"tools: {', '.join(tools)}")
                info = await session.call_tool("server_info", {})
                step(not info.is_error, f"server_info: {info.structured_content}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1].rstrip("/")))
