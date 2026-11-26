import json
from typing import NamedTuple
from urllib.parse import parse_qsl, urlencode, urlsplit

from cvat_sdk import Client
from cvat_sdk.api_client import models
from cvat_sdk.api_client.exceptions import ApiException
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument, explain
from cvat_mcp.services.tasks import clamp_limit
from cvat_mcp.services.users import require_own

S3 = "AWS_S3_BUCKET"  # CVAT's provider type for every S3-compatible host


def attributes(storage) -> dict[str, str]:
    """CVAT's specific_attributes, "endpoint_url=…&region=…&prefix=…", as a dict."""
    return dict(parse_qsl(storage.specific_attributes or ""))  # the UI percent-encodes them


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
        body = exc.body.decode(errors="replace") if isinstance(exc.body, bytes) else exc.body
        for secret in filter(None, secrets):  # before explain() cuts the body, splitting a key
            body = (body or "").replace(secret, "***")
        exc.body = body
        raise InvalidArgument(explain(exc)) from None  # no chained exception carrying the body


def _with_status(client: Client, storage, base_url: str) -> CloudStorageStatus:
    """The storage as CVAT stored it, and whether CVAT can reach it right now."""
    try:
        _, response = client.api_client.cloudstorages_api.retrieve_status(
            storage.id, _parse_response=False
        )
        status = json.loads(response.data)  # a bare JSON string, e.g. "AVAILABLE"
    except ApiException:
        status = "unknown"  # the write already happened: report it, don't fail
    return CloudStorageStatus(**storage_info(storage, base_url).model_dump(), status=status)


def _endpoint(endpoint_url: str | None) -> str | None:
    """The endpoint as given, without a trailing "/"; None or blank means AWS itself."""
    endpoint_url = (endpoint_url or "").strip()
    if not endpoint_url:  # models often send "" for an optional argument
        return None
    parts = urlsplit(endpoint_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise InvalidArgument(
            "endpoint_url must be an http(s) URL of the S3 API, e.g. https://<host>"
        )
    if parts.username or parts.password:
        raise InvalidArgument("don't put credentials in endpoint_url: pass the keys as arguments")
    if parts.path.strip("/") or parts.query or parts.fragment:  # e.g. R2's ".../<bucket>"
        raise InvalidArgument(
            "endpoint_url is the host only (https://<host>[:port]): pass the bucket as bucket "
            "and folders as cloud_path"
        )
    return endpoint_url.rstrip("/")


def _repeat(client: Client, bucket: str, endpoint: str | None):
    """A storage exactly like the one create_cloud_storage would make: whole bucket, personal."""
    page, _ = client.api_client.cloudstorages_api.list(
        resource=bucket, provider_type=S3, page_size=100
    )
    for storage in page.results:
        attrs = attributes(storage)
        if (
            storage.provider_type.value == S3
            and storage.resource == bucket
            and storage.organization is None  # tasks are created in the personal workspace
            and not attrs.get("prefix")
            and ((attrs.get("endpoint_url") or "").rstrip("/") or None) == endpoint
        ):
            return storage
    return None


def create_cloud_storage(
    client: Client,
    base_url: str,
    bucket: str,
    endpoint_url: str | None,
    region: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
    display_name: str | None,
) -> CloudStorageStatus:
    """Register a whole S3-compatible bucket; CVAT checks that it can reach it."""
    bucket = bucket.strip()
    if not bucket:
        raise InvalidArgument("bucket must not be empty")
    if any(c in bucket for c in "/:"):  # S3 bucket names have neither
        raise InvalidArgument(
            "bucket is the bucket name, not a link: pass the host as endpoint_url and choose "
            "folders per task with cloud_path"
        )
    endpoint = _endpoint(endpoint_url)
    credentials = _credentials(access_key_id, secret_access_key)
    if existing := _repeat(client, bucket, endpoint):
        raise InvalidArgument(
            f"bucket {bucket!r} is already registered as cloud storage #{existing.id}: use "
            "it, or change its keys with update_cloud_storage"
        )
    host = urlsplit(endpoint).netloc if endpoint else "AWS S3"
    attrs = {"endpoint_url": endpoint, "region": region}
    request = models.CloudStorageWriteRequest(
        provider_type=models.ProviderTypeEnum(S3),
        resource=bucket,
        display_name=display_name or f"{bucket} ({host})"[:63],  # CVAT's limit
        specific_attributes=urlencode({k: v for k, v in attrs.items() if v}),  # as the UI does
        **(credentials or {"credentials_type": models.CredentialsTypeEnum("ANONYMOUS_ACCESS")}),
    )
    api = client.api_client.cloudstorages_api
    storage, _ = _sent(
        lambda: api.create(request), credentials.get("key"), credentials.get("secret_key")
    )
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
    if storage.organization is not None:  # its tasks are outside the scan below
        raise InvalidArgument(
            f"cloud storage #{cloud_storage_id} belongs to an organization; delete it in the "
            "CVAT UI, where its tasks are visible"
        )
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


class CloudSource(NamedTuple):
    resources: list[str]  # CVAT's server_files: object keys
    filename_pattern: str | None


def cloud_source(key: str) -> CloudSource:
    """What a cloud_path selects: a glob, one object, or a folder (recursively)."""
    key = key.strip().lstrip("/")  # object keys never start with "/"
    if any(c in key for c in "*?"):  # not "[": keys like "shots[2024]/" contain it
        return CloudSource([], key)
    if "." in key.rsplit("/", 1)[-1]:  # a file name; "street" (no extension) is a folder
        return CloudSource([key], None)
    folder = key.rstrip("/")
    # CVAT expands "folder/" by listing only that prefix (recursively); a filename_pattern
    # makes it list the whole bucket first, which is slow on big buckets.
    return CloudSource([f"{folder}/"], None) if folder else CloudSource([], "*")


def check_storage(client: Client, storage_id: int, key: str) -> None:
    """Refuse a storage this account can't use for its personal-workspace tasks, or one whose
    prefix doesn't cover key (an object key, folder or glob from the bucket root)."""
    try:
        storage, _ = client.api_client.cloudstorages_api.retrieve(storage_id)
    except ApiException as exc:
        if exc.status not in (403, 404):
            raise
        raise InvalidArgument(
            f"no cloud storage #{storage_id} this account can use: find the bucket's id with "
            "list_cloud_storages"
        ) from None
    if storage.organization is not None:
        raise InvalidArgument(
            f"cloud storage #{storage_id} belongs to an organization, and this server works in "
            "the personal workspace: register the bucket there with create_cloud_storage"
        )
    prefix = attributes(storage).get("prefix", "")
    if not key.startswith(prefix):
        raise InvalidArgument(
            f"cloud storage #{storage_id} covers only {prefix!r} in its bucket: pass a cloud_path "
            "under it, or use a storage for the whole bucket (list_cloud_storages)"
        )


def object_exists(client: Client, storage_id: int, key: str) -> bool:
    """Whether the bucket has the object `key`, asked through CVAT (it holds the keys)."""
    name, token = key.rsplit("/", 1)[-1], None
    while True:
        # prefix=<full key> lists that object by its last segment, plus longer names
        page, _ = client.api_client.cloudstorages_api.retrieve_content_v2(
            storage_id, prefix=key, **({"next_token": token} if token else {})
        )
        if any(e.name == name and str(e.type) == "REG" for e in page.content):
            return True
        token = page.next
        if not token:
            return False


def keys_hint(storage_id: int, access: str = "read") -> str:
    return (
        f", and that cloud storage #{storage_id} can {access} the bucket (its keys: "
        "update_cloud_storage)"
    )
