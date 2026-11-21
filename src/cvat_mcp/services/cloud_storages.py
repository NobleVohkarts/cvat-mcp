import re
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

from cvat_sdk import Client
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument

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
        bucket, _, key = f"{host}/{path}".partition("/") if host else ("", "", "")
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


def find_storage(client: Client, ref: BucketRef, owner_id: int, *, whole_bucket: bool = False):
    """The CVAT storage serving this bucket and key: our own first, then the oldest."""
    page, _ = client.api_client.cloudstorages_api.list(resource=ref.bucket, page_size=100)
    found = [s for s in page.results if _matches(s, ref, whole_bucket)]
    return min(found, key=lambda s: (s.owner is None or s.owner.id != owner_id, s.id), default=None)
