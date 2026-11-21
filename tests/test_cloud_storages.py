import pytest
from cvat_sdk.api_client.exceptions import ApiException
from fakes import Obj, cloud_storage, fake_cloudstorages_api

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.cloud_storages import (
    BucketRef,
    CloudSource,
    cloud_source,
    create_cloud_storage,
    delete_cloud_storage,
    find_storage,
    list_cloud_storages,
    parse_bucket_url,
    resolve_storage,
    storage_info,
    update_cloud_storage,
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
        (
            f"https://photos.abc123.r2.cloudflarestorage.com/train/",
            (R2, "auto", "photos", "train/"),
        ),
        ("https://ABC123.R2.cloudflarestorage.com/cvat/", (R2, "auto", "cvat", "")),
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
        ("https://k:s@minio:9000/b/", "credentials"),
        ("https://b.s3-accelerate.amazonaws.com/x", r"s3://bucket/"),
        ("https://b.s3.dualstack.us-east-1.amazonaws.com/x", r"s3://bucket/"),
        ("https://b.s3-website-us-east-1.amazonaws.com/x", r"s3://bucket/"),
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
    # the CVAT UI percent-encodes the values (URLSearchParams)
    ui = storage_info(cloud_storage(3, "cvat", "endpoint_url=https%3A%2F%2Fe&prefix=a%2Fb%2F"), "")
    assert (ui.endpoint_url, ui.prefix) == ("https://e", "a/b/")


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
        (
            [
                cloud_storage(
                    1, "cvat", "endpoint_url=https%3A%2F%2Fabc123.r2.cloudflarestorage.com"
                )
            ],
            REF,
            {},
            1,
        ),
    ],
)
def test_find_storage(storages, ref, kw, expected):
    assert _find(storages, ref, **kw) == expected


ME = Obj(id=1, username="admin")
BASE = "http://cvat/"


def _crud_client(storages=(), tasks=()):
    """A client whose cloudstorages_api.create returns storage #7 built from the request."""
    client = _client(
        storages,
        users=Obj(retrieve_current_user=lambda: ME),
        tasks=Obj(list=lambda **_: list(tasks)),
    )
    api = client.api_client.cloudstorages_api
    api.partial_update.side_effect = lambda id, **_: api.retrieve(id)

    def create(request):
        sent = request.to_dict()
        storage = cloud_storage(
            7,
            sent["resource"],
            sent["specific_attributes"],
            credentials=sent["credentials_type"],
            name=sent["display_name"],
        )
        return storage, None

    api.create.side_effect = create
    return client, api


def _sent(mock):
    (request,), _ = mock.call_args
    return request.to_dict()


def test_list_newest_first_with_total():
    client, api = _crud_client([cloud_storage(2, "b"), cloud_storage(1, "cvat")])
    result = list_cloud_storages(client, BASE, "cv", 500)
    kwargs = api.list.call_args.kwargs
    assert (kwargs["sort"], kwargs["page_size"], kwargs["search"]) == ("-id", 100, "cv")
    assert [s.id for s in result.storages] == [2, 1] and result.total == 2


def test_create_with_keys_sends_s3_request():
    client, api = _crud_client()
    result = create_cloud_storage(client, BASE, f"{R2}/cvat/street/", "AK", "SK", None, None)
    assert _sent(api.create) == {
        "provider_type": "AWS_S3_BUCKET",
        "resource": "cvat",
        "display_name": "cvat (Cloudflare R2)",
        "credentials_type": "KEY_SECRET_KEY_PAIR",
        "key": "AK",
        "secret_key": "SK",
        "specific_attributes": "endpoint_url=https%3A%2F%2Fabc123.r2.cloudflarestorage.com"
        "&region=auto",
    }
    assert result.status == "AVAILABLE" and result.id == 7
    dumped = result.model_dump_json()
    assert "AK" not in dumped and "SK" not in dumped


def test_create_anonymous():
    client, api = _crud_client()
    create_cloud_storage(client, BASE, "s3://open-bucket/", None, None, None, None)
    sent = _sent(api.create)
    assert sent["credentials_type"] == "ANONYMOUS_ACCESS"
    assert "key" not in sent and "secret_key" not in sent
    assert (sent["specific_attributes"], sent["display_name"]) == ("", "open-bucket (AWS S3)")
    create_cloud_storage(client, BASE, "s3://open-bucket/", None, None, None, "us-east-1")
    assert _sent(api.create)["specific_attributes"] == "region=us-east-1"


def test_default_display_name_fits():
    client, api = _crud_client()
    create_cloud_storage(client, BASE, f"{R2}/{'b' * 63}/", "AK", "SK", None, None)
    assert len(_sent(api.create)["display_name"]) <= 63
    create_cloud_storage(client, BASE, "http://minio.local:9000/data/", "AK", "SK", None, None)
    assert _sent(api.create)["display_name"] == "data (minio.local:9000)"


@pytest.mark.parametrize(
    "url, key, secret, match",
    [
        (f"{R2}/cvat/", "AK", None, "both"),
        (f"{R2}/cvat/", "AK", "", "both"),
        (f"{R2}/cvat/", "AK", "SK", "already registered as cloud storage #1.*update_cloud_storage"),
        ("gs://cvat/", None, None, "only S3-compatible"),
    ],
)
def test_create_refusals_send_nothing(url, key, secret, match):
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2)])
    with pytest.raises(InvalidArgument, match=match):
        create_cloud_storage(client, BASE, url, key, secret, None, None)
    api.create.assert_not_called()


def test_error_never_contains_keys():
    client, api = _crud_client()
    refused = ApiException(status=400)
    refused.body = b'{"secret_key": "SK is invalid", "key": "AK"}'
    api.create.side_effect = refused
    with pytest.raises(InvalidArgument, match="CVAT rejected the request") as error:
        create_cloud_storage(client, BASE, f"{R2}/cvat/", "AK", "SK", None, None)
    message = str(error.value)
    assert "AK" not in message and "SK" not in message and "***" in message


def test_error_never_contains_part_of_a_key():
    """A key cut by the 300-character limit must not leave its first part behind."""
    client, api = _crud_client()
    secret = "SECRETKEYVALUE0123456789"
    refused = ApiException(status=400)
    refused.body = ("x" * 290 + secret).encode()  # the 300-character cut splits the key
    api.create.side_effect = refused
    with pytest.raises(InvalidArgument) as error:
        create_cloud_storage(client, BASE, f"{R2}/cvat/", "AKIDVALUE", secret, None, None)
    assert secret[:8] not in str(error.value)


def test_create_survives_status_failure():
    """The storage exists once create answers: a failing status read doesn't hide that."""
    client, api = _crud_client()
    api.retrieve_status.side_effect = ApiException(status=500, reason="Server Error")
    result = create_cloud_storage(client, BASE, f"{R2}/cvat/", "AK", "SK", None, None)
    assert (result.id, result.status) == (7, "unknown")


def test_update_rename_and_keys():
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2)])

    def sent_after(**changes):
        args = {"display_name": None, "access_key_id": None, "secret_access_key": None} | changes
        result = update_cloud_storage(client, BASE, 1, **args)
        assert result.status == "AVAILABLE"
        (storage_id,), kwargs = api.partial_update.call_args
        assert storage_id == 1
        return kwargs["patched_cloud_storage_write_request"].to_dict()

    assert sent_after(display_name="R2 street") == {"display_name": "R2 street"}
    assert sent_after(access_key_id="AK", secret_access_key="SK") == {
        "credentials_type": "KEY_SECRET_KEY_PAIR",
        "key": "AK",
        "secret_key": "SK",
    }
    assert sent_after(access_key_id="", secret_access_key="") == {
        "credentials_type": "ANONYMOUS_ACCESS",
        "key": "",
        "secret_key": "",
    }
    with pytest.raises(InvalidArgument, match="nothing to change"):
        sent_after()
    with pytest.raises(InvalidArgument, match="both"):
        sent_after(access_key_id="AK")


def test_delete_refused_while_tasks_use_it():
    tasks = [Obj(id=29, data_cloud_storage_id=1), Obj(id=30, data_cloud_storage_id=None)]
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2)], tasks)
    with pytest.raises(InvalidArgument, match="task 29.*delete or rebuild"):
        delete_cloud_storage(client, 1)
    api.destroy.assert_not_called()


def test_delete_own_unused():
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2, name="R2")])
    result = delete_cloud_storage(client, 1)
    api.destroy.assert_called_once_with(1)
    assert (result.cloud_storage_id, result.display_name) == (1, "R2")


def test_delete_organization_storage_refused():
    """Its tasks live in the organization, which this server's task scan can't see."""
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2, org=3)])
    with pytest.raises(InvalidArgument, match="organization.*CVAT UI"):
        delete_cloud_storage(client, 1)
    api.destroy.assert_not_called()


def test_delete_not_owned_refused():
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2, owner_id=2, owner="bob")])
    with pytest.raises(InvalidArgument, match="belongs to bob"):
        delete_cloud_storage(client, 1)
    api.destroy.assert_not_called()


@pytest.mark.parametrize(
    "key, expected",
    [
        ("", ([], "*")),
        ("street/", (["street/"], None)),
        ("street", (["street/"], None)),
        ("a/b", (["a/b/"], None)),
        ("street/*.jpg", ([], "street/*.jpg")),
        ("img_?.png", ([], "img_?.png")),
        ("shots[2024]/a.jpg", (["shots[2024]/a.jpg"], None)),  # [ is part of a key, not a glob
        ("street/000001.jpg", (["street/000001.jpg"], None)),
    ],
)
def test_cloud_source(key, expected):
    assert cloud_source(key) == CloudSource(*expected)


def test_resolve_reuses_match():
    client, api = _crud_client([cloud_storage(1, "cvat", AT_R2)])
    assert resolve_storage(client, parse_bucket_url(f"{R2}/cvat/street/")) == (1, False)
    api.create.assert_not_called()


def test_resolve_registers_anonymous():
    client, api = _crud_client()
    assert resolve_storage(client, parse_bucket_url("s3://open-bucket/x/")) == (7, True)
    assert _sent(api.create)["credentials_type"] == "ANONYMOUS_ACCESS"


def test_resolve_private_bucket_needs_keys():
    client, api = _crud_client()
    refused = ApiException(status=400)
    refused.body = b"The specified cloud storage is not available."
    api.create.side_effect = refused
    with pytest.raises(InvalidArgument, match="without keys") as error:
        resolve_storage(client, parse_bucket_url(f"{R2}/cvat/street/"))
    assert "create_cloud_storage(url, access_key_id, secret_access_key)" in str(error.value)
    assert "not available" in str(error.value)
