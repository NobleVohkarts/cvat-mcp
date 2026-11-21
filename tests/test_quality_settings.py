from unittest.mock import Mock

import pytest
from fakes import Obj, fake_client, fake_list_settings, fake_task
from fakes import quality_requirement as req

from cvat_mcp.errors import InvalidArgument
from cvat_mcp.services.quality_settings import settings_state, update_quality_settings

PROJECT_REQS = [req(154, "Base tag", "tag"), req(155, "Base rectangle", "rectangle")]
TASK_REQS = [req(253, "Base tag", "tag"), req(254, "Base rectangle", "rectangle")]


def _client(project_id=9, project_reqs=None, **settings_kw):
    quality_api = Obj(
        list_settings=fake_list_settings(
            21, list(TASK_REQS), project_reqs or list(PROJECT_REQS), **settings_kw
        ),
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
    assert state.shared_with_tasks == 2  # CVAT's count of inheriting tasks


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
    "client_kw, args, match",
    [
        ({}, (["Base box"], [], None), "unknown.*Base rectangle"),
        ({}, (["Base tag"], ["Base tag"], None), "both"),
        ({}, ([], [], None), "nothing to change"),
        ({"project_id": None}, ([], [], True), "not in a project"),
        ({"visible": False}, (["Base tag"], [], None), "not visible.*use_project_settings=false"),
    ],
)
def test_refusals_write_nothing(client_kw, args, match):
    client, _, api = _client(**client_kw)
    with pytest.raises(InvalidArgument, match=match):
        update_quality_settings(client, 21, *args)
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


def test_own_settings_skip_the_project():
    # switching to its own settings needs nothing of the project, which may be invisible
    client, _, api = _client(visible=False)
    result = update_quality_settings(client, 21, ["Base rectangle"], [], False)
    assert all(kw.get("parent_type") != "project" for _, kw in api.list_settings.call_args_list)
    assert result.source == "task"


def test_requirements_written_before_inherit():
    # a failed requirement write must not leave the task already switched
    client, _, api = _client(inherit=False)
    order = []
    api.partial_update_settings.side_effect = lambda *a, **k: order.append("inherit")
    api.partial_update_settings_requirements.side_effect = lambda *a, **k: order.append("req")
    result = update_quality_settings(client, 21, ["Base rectangle"], [], True)
    assert order == ["req", "inherit"]
    assert _patched_requirements(api) == [(155, True)]
    assert result.shared_with_tasks == 3  # counted before the switch, plus this task


def test_child_requirement_type_is_null():
    # CVAT stores no annotation type on child requirements (they take the parent's)
    reqs = [req(154, "Base rectangle", "rectangle"), req(157, "Strict boxes", None)]
    client, task, _ = _client(project_reqs=reqs)
    assert _state(settings_state(client, task))[1] == ("Strict boxes", None, False)
