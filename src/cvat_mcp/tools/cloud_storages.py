from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from cvat_mcp.server import get_client, settings, tool
from cvat_mcp.services import cloud_storages
from cvat_mcp.services.cloud_storages import (
    CloudStorageList,
    CloudStorageStatus,
    DeletedCloudStorage,
)

DisplayName = Annotated[str, Field(min_length=1, max_length=63, pattern=r"\S")]  # CVAT's limit


@tool()
def list_cloud_storages(
    ctx: Context, search: str | None = None, limit: int = 20
) -> CloudStorageList:
    """Use to find the cloud storages (buckets) CVAT knows: id, bucket, endpoint, prefix,
    whether it has keys (credentials_type), and owner; never the keys. Newest first, up to
    limit (max 100), plus total."""
    return cloud_storages.list_cloud_storages(get_client(ctx), settings.url, search, limit)


# open world: CVAT connects to the bucket's host
@tool(write=True, open_world_hint=True)
def create_cloud_storage(
    ctx: Context,
    bucket: str,
    endpoint_url: str | None = None,
    region: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    display_name: DisplayName | None = None,
) -> CloudStorageStatus:
    """Use to register an S3-compatible bucket in CVAT, only when the user asks: bucket is the
    bucket name; endpoint_url is the S3 API endpoint of a host other than AWS, host only (e.g.
    https://<account_id>.r2.cloudflarestorage.com for Cloudflare R2, or a MinIO or other
    S3-compatible server's URL), left out for AWS S3; region when the host needs one (an AWS
    bucket outside us-east-1, "auto" for R2). The whole bucket is registered; folders are chosen
    per task with cloud_path. Pass both keys for a private bucket, or neither for a public one;
    never repeat the keys to the user. CVAT checks that the bucket is reachable and refuses one
    it can't read. A bucket already registered with the same endpoint is refused with its id:
    use that id, or change its keys with update_cloud_storage."""
    return cloud_storages.create_cloud_storage(
        get_client(ctx),
        settings.url,
        bucket,
        endpoint_url,
        region,
        access_key_id,
        secret_access_key,
        display_name,
    )


# open world: CVAT re-checks the bucket's host on every update
@tool(write=True, idempotent_hint=True, open_world_hint=True)
def update_cloud_storage(
    ctx: Context,
    cloud_storage_id: int,
    display_name: DisplayName | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
) -> CloudStorageStatus:
    """Use to rename a cloud storage or replace its keys, only when the user asks. Both keys
    together; "" and "" switch to anonymous access. The bucket and endpoint can't change:
    register a new storage instead. Never repeat the keys to the user."""
    return cloud_storages.update_cloud_storage(
        get_client(ctx),
        settings.url,
        cloud_storage_id,
        display_name,
        access_key_id,
        secret_access_key,
    )


@tool(write=True, destructive_hint=True, idempotent_hint=True)
def delete_cloud_storage(ctx: Context, cloud_storage_id: int) -> DeletedCloudStorage:
    """Use to delete a cloud storage from CVAT (the bucket itself is untouched), only when the
    user explicitly asks. Only storages owned by this server's CVAT account, and only when no
    task's frames come from it (delete or rebuild those first). Tasks and projects that import
    or export through it switch back to local storage."""
    return cloud_storages.delete_cloud_storage(get_client(ctx), cloud_storage_id)
