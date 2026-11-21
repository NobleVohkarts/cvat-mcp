import pytest
from fakes import Obj, cloud_storage, fake_cloudstorages_api

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.cloud_storages import (
    BucketRef,
    find_storage,
    parse_bucket_url,
    storage_info,
)

R2 = "https://abc123.r2.cloudflarestorage.com"


@pytest.mark.parametrize(
    "url, expected",
    [
        ("s3://cvat/street/", (None, None, "cvat", "street/")),
        ("s3://cvat", (None, None, "cvat", "")),
        (
            "https://cvat.s3.eu-central-1.amazonaws.com/street/a.jpg",
            (None, "eu-central-1", "cvat", "street/a.jpg"),
        ),
        ("https://cvat.s3.amazonaws.com/x", (None, None, "cvat", "x")),
        (
            "https://s3.us-east-1.amazonaws.com/cvat/street/",
            (None, "us-east-1", "cvat", "street/"),
        ),
        (f"{R2}/cvat/street/", (R2, "auto", "cvat", "street/")),
        (f"{R2}/cvat", (R2, "auto", "cvat", "")),
        (
            "https://abc123.eu.r2.cloudflarestorage.com/cvat/",
            ("https://abc123.eu.r2.cloudflarestorage.com", "auto", "cvat", ""),
        ),
        (
            "http://minio.local:9000/data/img%20one.jpg",
            ("http://minio.local:9000", None, "data", "img one.jpg"),
        ),
    ],
)
def test_parse_bucket_url(url, expected):
    assert parse_bucket_url(url) == BucketRef(*expected)


@pytest.mark.parametrize(
    "url, match",
    [
        ("gs://b/x", "only S3-compatible"),
        ("https://acc.blob.core.windows.net/c/x", "only S3-compatible"),
        ("https://storage.googleapis.com/b/x", "only S3-compatible"),
        ("https://abc.r2.cloudflarestorage.com/", "no bucket"),
        ("s3://", "no bucket"),
        ("ftp://h/b", r"s3:// or http\(s\)"),
        ("https://h/b/a.jpg?X-Amz-Signature=1", "presigned"),
    ],
)
def test_parse_bucket_url_refused(url, match):
    with pytest.raises(InvalidArgument, match=match):
        parse_bucket_url(url)


def test_storage_info_parses_attributes():
    info = storage_info(
        cloud_storage(1, "cvat", "endpoint_url=https://e&region=auto&prefix=street/"),
        "http://cvat/",
    )
    assert (info.endpoint_url, info.region, info.prefix) == ("https://e", "auto", "street/")
    assert (info.id, info.bucket, info.provider) == (1, "cvat", "AWS_S3_BUCKET")
    assert (info.credentials_type, info.owner) == ("KEY_SECRET_KEY_PAIR", "admin")
    assert info.url == "http://cvat/cloudstorages/update/1"
    bare = storage_info(cloud_storage(2, "cvat"), "http://cvat/")
    assert (bare.endpoint_url, bare.region, bare.prefix) == (None, None, None)


def _client(storages=(), **repos):
    return Obj(api_client=Obj(cloudstorages_api=fake_cloudstorages_api(storages)), **repos)


def _find(storages, ref, owner_id=1, **kw):
    client = _client(storages)
    found = find_storage(client, ref, owner_id, **kw)
    assert client.api_client.cloudstorages_api.list.call_args.kwargs["resource"] == ref.bucket
    return found and found.id


REF = BucketRef(R2, "auto", "cvat", "street/a.jpg")
AT_R2 = f"endpoint_url={R2}&region=auto"


@pytest.mark.parametrize(
    "storages, ref, kw, expected",
    [
        ([cloud_storage(1, "cvat", "endpoint_url=https://other&region=auto")], REF, {}, None),
        ([cloud_storage(1, "cvat", f"endpoint_url={R2}/&region=auto")], REF, {}, 1),
        ([cloud_storage(1, "cvat", AT_R2, provider="GOOGLE_CLOUD_STORAGE")], REF, {}, None),
        ([cloud_storage(1, "cvat", AT_R2, org=5)], REF, {}, None),
        (
            [cloud_storage(1, "cvat", AT_R2, owner_id=2), cloud_storage(3, "cvat", AT_R2)],
            REF,
            {},
            3,
        ),
        (
            [
                cloud_storage(4, "cvat", AT_R2, owner_id=2),
                cloud_storage(3, "cvat", AT_R2, owner_id=2),
            ],
            REF,
            {},
            3,
        ),
        ([cloud_storage(1, "cvat", f"{AT_R2}&prefix=street/")], REF, {}, 1),
        (
            [cloud_storage(1, "cvat", f"{AT_R2}&prefix=street/")],
            REF._replace(key="roads/"),
            {},
            None,
        ),
        (
            [cloud_storage(1, "cvat", f"{AT_R2}&prefix=street/")],
            REF,
            {"whole_bucket": True},
            None,
        ),
        ([cloud_storage(1, "cvat")], BucketRef(None, None, "cvat", ""), {}, 1),
    ],
)
def test_find_storage(storages, ref, kw, expected):
    assert _find(storages, ref, **kw) == expected
