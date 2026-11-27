import logging
import os
import sys

from pydantic import ValidationError

from cvat_mcp.config import Settings


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # stdout is the MCP channel
    logging.getLogger("cvat_sdk").setLevel(logging.WARNING)

    # Imported here so a bad configuration becomes a one-line message instead of a traceback.
    try:
        import cvat_mcp.tools  # pylint: disable=import-outside-toplevel,unused-import
        from cvat_mcp.server import (  # pylint: disable=import-outside-toplevel
            HTTP_MODE,
            check_cvat,
            mcp,
            settings,
            transport_security,
        )
    except ValidationError as e:
        prefix = Settings.model_config["env_prefix"]
        names = ", ".join(f"{prefix}{str(err['loc'][0]).upper()}" for err in e.errors())
        sys.exit(f"cvat-mcp: invalid or missing {names}. See .env.example.")

    try:
        if HTTP_MODE:
            if "*" in (h.strip() for h in os.environ.get("FORWARDED_ALLOW_IPS", "").split(",")):
                sys.exit(
                    "cvat-mcp: FORWARDED_ALLOW_IPS=* lets any client name its own address and "
                    "dodge the login lockout; list the proxies' addresses instead."
                )
            # The web server swallows exits from its lifespan, so check CVAT before serving.
            check_cvat()
            # Bound to localhost only: a tunnel or reverse proxy on this machine is the one way in.
            mcp.run(
                "streamable-http",
                host="127.0.0.1",
                port=settings.mcp_port,
                transport_security=transport_security(),
            )
        else:
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
