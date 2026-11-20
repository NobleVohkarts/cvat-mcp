import logging
import sys

from pydantic import ValidationError

from cvat_mcp.config import Settings


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # stdout is the MCP channel
    logging.getLogger("cvat_sdk").setLevel(logging.WARNING)

    # Imported here so a bad configuration becomes a one-line message instead of a traceback.
    try:
        import cvat_mcp.tools  # pylint: disable=import-outside-toplevel,unused-import
        from cvat_mcp.server import mcp  # pylint: disable=import-outside-toplevel
    except ValidationError as e:
        prefix = Settings.model_config["env_prefix"]
        names = ", ".join(f"{prefix}{str(err['loc'][0]).upper()}" for err in e.errors())
        sys.exit(f"cvat-mcp: invalid or missing {names}. See cvat-mcp/.env.example.")

    try:
        mcp.run("stdio")
    except BaseExceptionGroup as group:
        # anyio wraps a startup failure in the lifespan; show its one-line message.
        exits, rest = group.split(SystemExit)
        if exits is None or rest is not None:
            raise
        while isinstance(exits, BaseExceptionGroup):
            exits = exits.exceptions[0]
        sys.exit(exits.code)


if __name__ == "__main__":
    main()
