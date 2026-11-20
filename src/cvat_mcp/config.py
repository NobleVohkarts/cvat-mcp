from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from cvat_mcp.errors import InvalidArgument


class Settings(BaseSettings):
    """Server configuration from environment variables (or `.env`). The server never stores a
    password; in HTTP mode its login page checks the PAT owner's CVAT password once."""

    model_config = SettingsConfigDict(env_prefix="CVAT_", env_file=".env", extra="ignore")

    url: str = "http://localhost:8080"
    access_token: SecretStr
    mcp_mode: Literal["read-only", "write"] = "read-only"
    mcp_workdir: Path = Path("workdir")
    # HTTP mode for remote clients (claude.ai) through a tunnel; stdio needs none of these.
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_public_url: str | None = None
    mcp_port: int = 8765
    mcp_redirect_hosts: str = "claude.ai,claude.com,localhost,127.0.0.1"

    @model_validator(mode="after")
    def _check_http_mode(self) -> "Settings":
        if self.mcp_transport == "http":
            url = urlsplit(self.mcp_public_url or "")
            local = url.hostname in ("localhost", "127.0.0.1")
            if not url.hostname or (url.scheme != "https" and not (local and url.scheme == "http")):
                raise ValueError(
                    "mcp_public_url (CVAT_MCP_PUBLIC_URL) must be the https URL clients use "
                    "(http only for localhost) when CVAT_MCP_TRANSPORT=http"
                )
        return self

    @property
    def public_url(self) -> str:
        return (self.mcp_public_url or "").rstrip("/")

    @property
    def redirect_hosts(self) -> set[str]:
        return {h.strip() for h in self.mcp_redirect_hosts.split(",") if h.strip()}


def resolve_in_workdir(workdir: Path, path: str | Path) -> Path:
    """The only way to turn a path from the model into a file path: it must stay in the workdir."""
    root = workdir.resolve()
    message = f"path must name a file inside CVAT_MCP_WORKDIR ({root})"
    try:
        resolved = (root / path).resolve()  # follows symlinks and "..", so both are caught
    except (OSError, ValueError) as exc:  # e.g. an embedded NUL byte
        raise InvalidArgument(message) from exc
    if resolved == root or not resolved.is_relative_to(root):
        raise InvalidArgument(message)
    return resolved
