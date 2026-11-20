import sys


def main() -> None:
    from cvat_mcp.server import mcp  # pylint: disable=import-outside-toplevel

    import cvat_mcp.tools  # pylint: disable=import-outside-toplevel,unused-import

    try:
        mcp.run("stdio")
    except BaseExceptionGroup as group:
        # A startup failure in the lifespan arrives wrapped by anyio; show its one-line message.
        exits, rest = group.split(SystemExit)
        if exits is None or rest is not None:
            raise
        while isinstance(exits, BaseExceptionGroup):
            exits = exits.exceptions[0]
        sys.exit(exits.code)


if __name__ == "__main__":
    main()
