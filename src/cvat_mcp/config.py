from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from cvat_mcp.errors import InvalidArgument


class Settings(BaseSettings):
    """Server configuration. Only environment variables (or `.env`); passwords are not supported."""

    model_config = SettingsConfigDict(env_prefix="CVAT_", env_file=".env", extra="ignore")

    url: str = "http://localhost:8080"
    access_token: SecretStr
    mcp_mode: Literal["read-only", "write"] = "read-only"
    mcp_workdir: Path = Path("workdir")


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
