from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOCAL_HOSTS = {"localhost", "127.0.0.1"}


def is_secure_url(url: str, hosts: set[str] | None = None) -> bool:
    """https (http only for a local host), no query or fragment, host in `hosts` if given."""
    parts = urlsplit(url)
    if not parts.hostname or parts.query or parts.fragment:
        return False
    if hosts is not None and parts.hostname not in hosts:
        return False
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOCAL_HOSTS)


class Settings(BaseSettings):
    """Server configuration from environment variables (or `.env`). The server never stores a
    password; in HTTP mode its login page checks the PAT owner's CVAT password once."""

    model_config = SettingsConfigDict(env_prefix="CVAT_", env_file=".env", extra="ignore")

    url: str = "http://localhost:8080"
    access_token: SecretStr
    mcp_mode: Literal["read-only", "write"] = "read-only"
    # HTTP mode for remote clients (claude.ai) behind a tunnel or reverse proxy; stdio needs none.
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_public_url: str | None = Field(default=None, validate_default=True)
    mcp_port: int = 8765
    mcp_redirect_hosts: str = "claude.ai,claude.com,localhost,127.0.0.1"

    @field_validator("mcp_public_url")
    @classmethod
    def _check_public_url(cls, url: str | None, info: ValidationInfo) -> str | None:
        if info.data.get("mcp_transport") == "http" and (
            not is_secure_url(url or "") or urlsplit(url).path not in ("", "/")
        ):
            raise ValueError(
                "must be the server's https base URL, without a path (http only for localhost)"
            )
        return url

    @property
    def public_url(self) -> str:
        return (self.mcp_public_url or "").rstrip("/")

    @property
    def redirect_hosts(self) -> set[str]:
        return {h.strip() for h in self.mcp_redirect_hosts.split(",") if h.strip()}
