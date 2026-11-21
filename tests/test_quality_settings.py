from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client, fake_task

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.quality_settings import settings_state, update_quality_settings


def _req(id_, name, annotation_type, enabled=False):
    return Obj(id=id_, name=name, annotation_type=annotation_type, enabled=enabled)


def _settings(id_, task_id, inherit, requirements, project_id=None):
    return Obj(
        id=id_, task_id=task_id, project_id=project_id, inherit=inherit, requirements=requirements
    )


def _client(project_id=9, inherit=True, task_reqs=None, project_reqs=None):
    task_settings = _settings(
        30,
        21,
        inherit,
        task_reqs or [_req(253, "Base tag", "tag"), _req(254, "Base rectangle", "rectangle")],
    )
    project_settings = _settings(
        18,
        None,
        True,
        project_reqs or [_req(154, "Base tag", "tag"), _req(155, "Base rectangle", "rectangle")],
        project_id=9,
    )
    other_task = _settings(19, 10, True, [])

    def list_settings(task_id=None, project_id=None, **_):
        if task_id is not None:
            return Obj(results=[task_settings]), None
        return Obj(results=[project_settings, other_task, task_settings]), None

    quality_api = Obj(
        list_settings=Mock(side_effect=list_settings),
        partial_update_settings=Mock(),
        partial_update_settings_requirements=Mock(),
    )
    task = fake_task(task_id=21, project_id=project_id)
    return fake_client(task, api_client=Obj(quality_api=quality_api)), task, quality_api


def _patched_requirements(api):
    return [
        (args[0], kwargs["patched_quality_requirement_request"].enabled)
        for args, kwargs in api.partial_update_settings_requirements.call_args_list
    ]


def _state(result):
    return [(r.name, r.annotation_type, r.enabled) for r in result.requirements]


def test_state_inherited_from_project():
    client, task, _ = _client()
    state = settings_state(client, task)
    assert (state.task_id, state.source, state.project_id) == (21, "project", 9)
    assert _state(state) == [("Base tag", "tag", False), ("Base rectangle", "rectangle", False)]
    assert state.shared_with_tasks == 2


def test_state_own_without_project():
    client, task, api = _client(project_id=None)
    state = settings_state(client, task)
    assert (state.source, state.project_id, state.shared_with_tasks) == ("task", None, None)
    assert all("project_id" not in kw for _, kw in api.list_settings.call_args_list)


def test_enable_on_project():
    client, _, api = _client()
    result = update_quality_settings(client, 21, ["Base rectangle"], [], None)
    assert _patched_requirements(api) == [(155, True)]
    api.partial_update_settings.assert_not_called()
    assert result.source == "project"
    assert ("Base rectangle", "rectangle", True) in _state(result)


def test_own_settings_then_enable():
    client, _, api = _client()
    result = update_quality_settings(client, 21, ["Base rectangle"], [], False)
    (settings_id,), kwargs = api.partial_update_settings.call_args
    assert (settings_id, kwargs["patched_quality_settings_request"].inherit) == (30, False)
    assert _patched_requirements(api) == [(254, True)]
    assert (result.source, result.shared_with_tasks) == ("task", None)
    assert ("Base rectangle", "rectangle", True) in _state(result)


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"enable": ["Base box"]}, "unknown.*Base rectangle"),
        ({"enable": ["Base tag"], "disable": ["Base tag"]}, "both"),
        ({}, "nothing to change"),
        ({"use_project_settings": True, "project_id": None}, "not in a project"),
        ({"enable": ["Base tag"], "dup": True}, "ambiguous"),
    ],
)
def test_refusals_write_nothing(kw, match):
    reqs = [_req(154, "Base tag", "tag"), _req(156, "Base tag", "tag")] if kw.get("dup") else None
    client, _, api = _client(project_id=kw.get("project_id", 9), project_reqs=reqs)
    with pytest.raises(InvalidArgument, match=match):
        update_quality_settings(
            client,
            21,
            kw.get("enable", []),
            kw.get("disable", []),
            kw.get("use_project_settings"),
        )
    api.partial_update_settings.assert_not_called()
    api.partial_update_settings_requirements.assert_not_called()


def test_no_project_own_settings_is_noop():
    client, _, api = _client(project_id=None)
    result = update_quality_settings(client, 21, [], [], False)
    api.partial_update_settings.assert_not_called()
    assert result.source == "task"


def test_already_in_state_skipped():
    client, _, api = _client()
    result = update_quality_settings(client, 21, [], ["Base tag"], None)
    api.partial_update_settings_requirements.assert_not_called()
    assert ("Base tag", "tag", False) in _state(result)
