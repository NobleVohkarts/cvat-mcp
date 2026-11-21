from cvat_sdk import Client

from cvat_mcp.errors import InvalidArgument


def user_id(client: Client, username: str) -> int | None:
    """The CVAT user id for a username; "" unassigns (None)."""
    if username == "":
        return None
    user = next((u for u in client.users.list(username=username) if u.username == username), None)
    if user is None:
        raise InvalidArgument(f'no CVAT user {username!r}; pass an existing username or ""')
    return user.id


def is_own(obj, owner_id: int) -> bool:
    """Owned by this server's CVAT user: what it may delete."""
    return obj.owner is not None and obj.owner.id == owner_id


def require_own(client: Client, obj, what: str) -> None:
    """Refuse to delete what this server's user doesn't own; others are handled in the UI."""
    me = client.users.retrieve_current_user()
    if not is_own(obj, me.id):
        owner = obj.owner.username if obj.owner else "a deleted user"
        raise InvalidArgument(
            f"{what} belongs to {owner}, not to this server's user {me.username}; only what it "
            "owns can be deleted here, the rest in the CVAT UI"
        )
