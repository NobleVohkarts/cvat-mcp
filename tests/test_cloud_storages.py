from unittest.mock import Mock

import pytest
from cvat_sdk.api_client.exceptions import ApiException
from fakes import Obj, cloud_storage, fake_cloudstorages_api

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.cloud_storages import (
    CloudSource,
    cloud_source,
    create_cloud_storage,
    delete_cloud_storage,
    list_cloud_storages,
    object_exists,
    storage_info,
    update_cloud_storage,
)

R2 = "https://abc123.r2.cloudflarestorage.com"


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


AT_R2 = f"endpoint_url={R2}&region=auto"


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


def _new(client, bucket="cvat", endpoint_url=R2, region=None, key="AK", secret="SK", name=None):
    return create_cloud_storage(client, BASE, bucket, endpoint_url, region, key, secret, name)


def test_create_with_keys_sends_s3_request():
    client, api = _crud_client()
    result = _new(client, region="auto")
    assert _sent(api.create) == {
        "provider_type": "AWS_S3_BUCKET",
        "resource": "cvat",
        "display_name": "cvat (abc123.r2.cloudflarestorage.com)",
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
    _new(client, "open-bucket", None, key=None, secret=None)
    sent = _sent(api.create)
    assert sent["credentials_type"] == "ANONYMOUS_ACCESS"
    assert "key" not in sent and "secret_key" not in sent
    assert (sent["specific_attributes"], sent["display_name"]) == ("", "open-bucket (AWS S3)")
    _new(client, "open-bucket", None, "us-east-1", key=None, secret=None)
    assert _sent(api.create)["specific_attributes"] == "region=us-east-1"


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_endpoint_means_aws(blank):
    client, api = _crud_client()
    _new(client, "open-bucket", blank, key=None, secret=None)
    sent = _sent(api.create)
    assert (sent["specific_attributes"], sent["display_name"]) == ("", "open-bucket (AWS S3)")


def test_endpoint_used_as_given():
    client, api = _crud_client()
    _new(client, "data", "http://minio.local:9000/")
    sent = _sent(api.create)
    assert sent["specific_attributes"] == "endpoint_url=http%3A%2F%2Fminio.local%3A9000"
    assert sent["display_name"] == "data (minio.local:9000)"
    _new(client, "b" * 63)
    assert len(_sent(api.create)["display_name"]) <= 63


@pytest.mark.parametrize("endpoint", [R2, R2 + "/"])
@pytest.mark.parametrize(
    "stored", [AT_R2, "endpoint_url=https%3A%2F%2Fabc123.r2.cloudflarestorage.com%2F"]
)
def test_create_refuses_exact_repeat(endpoint, stored):
    client, api = _crud_client([cloud_storage(1, "cvat", stored)])
    with pytest.raises(
        InvalidArgument, match="already registered as cloud storage #1.*update_cloud_storage"
    ):
        _new(client, endpoint_url=endpoint)
    api.create.assert_not_called()


@pytest.mark.parametrize(
    "existing",
    [
        cloud_storage(1, "cvat"),  # the same name on AWS is another bucket
        cloud_storage(1, "cvat", f"{AT_R2}&prefix=street/"),
        cloud_storage(1, "cvat", AT_R2, org=5),
        cloud_storage(1, "cvat", AT_R2, provider="GOOGLE_CLOUD_STORAGE"),
        cloud_storage(1, "other", AT_R2),
    ],
)
def test_create_not_a_repeat(existing):
    client, _ = _crud_client([existing])
    assert _new(client).id == 7


@pytest.mark.parametrize(
    "args, match",
    [
        ({"key": "AK", "secret": None}, "both"),
        ({"key": "AK", "secret": ""}, "both"),
        ({"bucket": " "}, "bucket"),
        ({"bucket": "s3://cvat/street"}, "not a link"),
        ({"bucket": "cvat/street"}, "not a link"),
        ({"endpoint_url": "https://AK:SK@e.com"}, "credentials"),
        ({"endpoint_url": "ftp://e.com"}, "http"),
        ({"endpoint_url": "https://"}, "http"),
        ({"endpoint_url": f"{R2}/cvat"}, "host only"),  # the R2 dashboard shows this form
        ({"endpoint_url": "https://e.com/?x=1"}, "host only"),
    ],
)
def test_create_refusals_send_nothing(args, match):
    client, api = _crud_client()
    with pytest.raises(InvalidArgument, match=match) as error:
        _new(client, **args)
    api.create.assert_not_called()
    assert "SK" not in str(error.value)


def test_error_never_contains_keys():
    client, api = _crud_client()
    refused = ApiException(status=400)
    refused.body = b'{"secret_key": "SK is invalid", "key": "AK"}'
    api.create.side_effect = refused
    with pytest.raises(InvalidArgument, match="CVAT rejected the request") as error:
        _new(client)
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
        _new(client, key="AKIDVALUE", secret=secret)
    assert secret[:8] not in str(error.value)


def test_create_survives_status_failure():
    """The storage exists once create answers: a failing status read doesn't hide that."""
    client, api = _crud_client()
    api.retrieve_status.side_effect = ApiException(status=500, reason="Server Error")
    result = _new(client)
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
        (" /street/ ", (["street/"], None)),  # keys never start with "/"
        ("/annotations/a.json", (["annotations/a.json"], None)),
        ("/", ([], "*")),
    ],
)
def test_cloud_source(key, expected):
    assert cloud_source(key) == CloudSource(*expected)


def _listing(*pages):
    """retrieve_content_v2 returning `pages` in turn; each page is a list of (name, type)."""
    replies = [
        (Obj(content=[Obj(name=n, type=t) for n, t in page], next=str(i + 1) if i + 1 < len(pages) else None), None)
        for i, page in enumerate(pages)
    ]
    return Mock(side_effect=replies)


def _content_client(listing):
    return Obj(api_client=Obj(cloudstorages_api=Obj(retrieve_content_v2=listing)))


def test_object_exists_matches_the_last_key_segment():
    listing = _listing([("a.zip", "REG")])
    assert object_exists(_content_client(listing), 1, "exports/a.zip")
    assert listing.call_args.args == (1,)
    assert listing.call_args.kwargs["prefix"] == "exports/a.zip"


def test_object_exists_ignores_longer_names_and_folders():
    listing = _listing([("a.zip.bak", "REG"), ("a.zip", "DIR")])
    assert not object_exists(_content_client(listing), 1, "exports/a.zip")


def test_object_exists_follows_pages():
    listing = _listing([("a.zip.bak", "REG")], [("a.zip", "REG")])
    assert object_exists(_content_client(listing), 1, "exports/a.zip")
    assert listing.call_args.kwargs["next_token"] == "1"
