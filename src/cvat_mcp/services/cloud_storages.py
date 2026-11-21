import json
import re
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.api_client.exceptions import ApiException
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain
from cvat_mcp.services.tasks import clamp_limit
from cvat_mcp.services.users import is_own, require_own

S3 = "AWS_S3_BUCKET"
NOT_S3 = (
    "only S3-compatible buckets (AWS S3, Cloudflare R2, MinIO) for now; add others in the CVAT UI"
)
AWS_VIRTUAL = re.compile(r"^(?P<bucket>.+)\.s3(?:[.-](?P<region>[a-z0-9-]+))?\.amazonaws\.com$")
AWS_PATH = re.compile(r"^s3(?:[.-](?P<region>[a-z0-9-]+))?\.amazonaws\.com$")
R2_SUFFIX = ".r2.cloudflarestorage.com"


class BucketRef(NamedTuple):
    endpoint_url: str | None  # None: AWS itself
    region: str | None
    bucket: str
    key: str


def parse_bucket_url(url: str) -> BucketRef:
    """A bucket link (s3://, AWS, R2 or another S3-compatible host) split into its parts."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.scheme == "gs" or host.endswith((".blob.core.windows.net", "storage.googleapis.com")):
        raise InvalidArgument(NOT_S3)
    if parts.scheme not in ("s3", "http", "https"):
        raise InvalidArgument(f"{url!r} is not a bucket link: pass s3:// or http(s)://")
    if parts.query or parts.fragment:
        raise InvalidArgument(
            "a link with a query looks like a presigned URL: pass it in urls, or the bucket link"
        )
    path = unquote(parts.path).lstrip("/")
    endpoint, region = None, None
    if parts.scheme == "s3":
        bucket, key = host, path
    elif m := AWS_VIRTUAL.match(host):
        bucket, key, region = m["bucket"], path, m["region"]
    else:
        bucket, _, key = path.partition("/")
        if m := AWS_PATH.match(host):
            region = m["region"]
        else:
            endpoint = f"{parts.scheme}://{parts.netloc}"
            region = "auto" if host.endswith(R2_SUFFIX) else None
    if not bucket:
        raise InvalidArgument(f"{url!r} names no bucket: pass a link like s3://bucket/folder/")
    return BucketRef(endpoint, region, bucket, key)


def attributes(storage) -> dict[str, str]:
    """CVAT's specific_attributes, "endpoint_url=…&region=…&prefix=…", as a dict."""
    return dict(kv.split("=", 1) for kv in (storage.specific_attributes or "").split("&") if kv)


class CloudStorageInfo(BaseModel):
    id: int
    display_name: str
    provider: str
    bucket: str
    endpoint_url: str | None
    region: str | None
    prefix: str | None
    credentials_type: str
    owner: str | None
    url: str


def storage_info(storage, base_url: str) -> CloudStorageInfo:
    attrs = attributes(storage)
    return CloudStorageInfo(
        id=storage.id,
        display_name=storage.display_name,
        provider=storage.provider_type.value,
        bucket=storage.resource,
        endpoint_url=attrs.get("endpoint_url"),
        region=attrs.get("region"),
        prefix=attrs.get("prefix"),
        credentials_type=storage.credentials_type.value,
        owner=storage.owner.username if storage.owner else None,
        url=f"{base_url.rstrip('/')}/cloudstorages/update/{storage.id}",
    )


def _matches(storage, ref: BucketRef, whole_bucket: bool) -> bool:
    attrs = attributes(storage)
    endpoint = (attrs.get("endpoint_url") or "").rstrip("/") or None
    prefix = attrs.get("prefix", "")
    return (
        storage.provider_type.value == S3
        and storage.resource == ref.bucket
        and storage.organization is None  # tasks are created in the personal workspace
        and endpoint == ref.endpoint_url
        and (not prefix if whole_bucket else ref.key.startswith(prefix))
    )


def find_storage(
    client: Client, ref: BucketRef, owner_id: int | None, *, whole_bucket: bool = False
):
    """The CVAT storage serving this bucket and key: our own first, then the oldest."""
    page, _ = client.api_client.cloudstorages_api.list(
        resource=ref.bucket, provider_type=S3, page_size=100
    )
    found = [s for s in page.results if _matches(s, ref, whole_bucket)]
    return min(found, key=lambda s: (not is_own(s, owner_id), s.id), default=None)


class CloudStorageList(BaseModel):
    storages: list[CloudStorageInfo]
    total: int


class CloudStorageStatus(CloudStorageInfo):
    status: str


class DeletedCloudStorage(BaseModel):
    cloud_storage_id: int
    display_name: str


def list_cloud_storages(
    client: Client, base_url: str, search: str | None, limit: int
) -> CloudStorageList:
    kwargs = {"search": search} if search else {}
    page, _ = client.api_client.cloudstorages_api.list(
        sort="-id", page_size=clamp_limit(limit), **kwargs
    )
    return CloudStorageList(
        storages=[storage_info(s, base_url) for s in page.results], total=page.count
    )


def _credentials(access_key_id: str | None, secret_access_key: str | None) -> dict:
    """The request fields for a key pair; {} when neither is given."""
    if access_key_id is None and secret_access_key is None:
        return {}
    if access_key_id is None or secret_access_key is None:
        raise InvalidArgument("pass both access_key_id and secret_access_key, or neither")
    if bool(access_key_id) != bool(secret_access_key):
        raise InvalidArgument('pass both keys, or both as "" for anonymous access')
    kind = "KEY_SECRET_KEY_PAIR" if access_key_id else "ANONYMOUS_ACCESS"
    return {
        "credentials_type": models.CredentialsTypeEnum(kind),
        "key": access_key_id,
        "secret_key": secret_access_key,
    }


def _sent(call, *secrets: str | None):
    """Run a create/update request; CVAT's refusal comes back without the keys in it."""
    try:
        return call()
    except ApiException as exc:
        if exc.status != 400:
            raise  # explain() shows no body for these
        message = explain(exc)
        for secret in filter(None, secrets):
            message = message.replace(secret, "***")
        raise InvalidArgument(message) from None  # no chained exception carrying the body


def _with_status(client: Client, storage, base_url: str) -> CloudStorageStatus:
    _, response = client.api_client.cloudstorages_api.retrieve_status(
        storage.id, _parse_response=False
    )
    status = json.loads(response.data)  # a bare JSON string, e.g. "AVAILABLE"
    return CloudStorageStatus(**storage_info(storage, base_url).model_dump(), status=status)


def _label(ref: BucketRef) -> str:
    if ref.endpoint_url is None:
        return "AWS S3"
    host = urlsplit(ref.endpoint_url).netloc
    return "Cloudflare R2" if host.endswith(R2_SUFFIX) else host


def create_cloud_storage(
    client: Client,
    base_url: str,
    url: str,
    access_key_id: str | None,
    secret_access_key: str | None,
    display_name: str | None,
    region: str | None,
) -> CloudStorageStatus:
    """Register a whole S3-compatible bucket; CVAT checks that it can reach it."""
    ref = parse_bucket_url(url)
    credentials = _credentials(access_key_id, secret_access_key) or {
        "credentials_type": models.CredentialsTypeEnum("ANONYMOUS_ACCESS")
    }
    if existing := find_storage(client, ref, None, whole_bucket=True):
        raise InvalidArgument(
            f"bucket {ref.bucket!r} is already registered as cloud storage #{existing.id}: use "
            "it, or change its keys with update_cloud_storage"
        )
    attrs = {"endpoint_url": ref.endpoint_url, "region": region or ref.region}
    request = models.CloudStorageWriteRequest(
        provider_type=models.ProviderTypeEnum(S3),
        resource=ref.bucket,
        display_name=display_name or f"{ref.bucket} ({_label(ref)})"[:63],  # CVAT's limit
        specific_attributes="&".join(f"{k}={v}" for k, v in attrs.items() if v),
        **credentials,
    )
    api = client.api_client.cloudstorages_api
    storage, _ = _sent(lambda: api.create(request), access_key_id, secret_access_key)
    return _with_status(client, storage, base_url)


def update_cloud_storage(
    client: Client,
    base_url: str,
    cloud_storage_id: int,
    display_name: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
) -> CloudStorageStatus:
    changes = _credentials(access_key_id, secret_access_key)
    if display_name is not None:
        changes["display_name"] = display_name
    if not changes:
        raise InvalidArgument("nothing to change: pass display_name or both keys")
    api = client.api_client.cloudstorages_api
    request = models.PatchedCloudStorageWriteRequest(**changes)
    storage, _ = _sent(
        lambda: api.partial_update(cloud_storage_id, patched_cloud_storage_write_request=request),
        access_key_id,
        secret_access_key,
    )
    return _with_status(client, storage, base_url)


def delete_cloud_storage(client: Client, cloud_storage_id: int) -> DeletedCloudStorage:
    """Delete our own storage, unless some task's frames still come from it."""
    api = client.api_client.cloudstorages_api
    storage, _ = api.retrieve(cloud_storage_id)
    require_own(client, storage, f"cloud storage #{cloud_storage_id}")
    # shortcut: scans all tasks, upgrade if CVAT adds a data_cloud_storage_id filter
    tasks = client.tasks.list(page_size=500)  # CVAT's maximum; its default is 10
    users = [t.id for t in tasks if t.data_cloud_storage_id == cloud_storage_id]
    if users:
        listed = ", ".join(f"task {i}" for i in users)
        raise InvalidArgument(
            f"cloud storage #{cloud_storage_id} still holds the frames of {listed}: delete or "
            "rebuild those tasks first"
        )
    api.destroy(cloud_storage_id)
    return DeletedCloudStorage(cloud_storage_id=cloud_storage_id, display_name=storage.display_name)
