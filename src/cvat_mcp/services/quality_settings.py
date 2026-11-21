from typing import Literal

from cvat_sdk import Client
from cvat_sdk.api_client import models
from pydantic import BaseModel

from cvat_mcp.errors import InvalidArgument


class Requirement(BaseModel):
    name: str
    annotation_type: str
    enabled: bool


class QualitySettings(BaseModel):
    task_id: int
    source: Literal["task", "project"]  # where the task's effective requirements live
    project_id: int | None
    requirements: list[Requirement]
    shared_with_tasks: int | None  # project source: tasks whose quality these settings decide


def settings_state(client: Client, task) -> QualitySettings:
    """The task's effective quality requirements and where they live."""
    own, rows = _load(client.api_client.quality_api, task)
    target, shared = _target(rows, own, own.inherit)
    return _result(task, target, shared, {})


def update_quality_settings(
    client: Client,
    task_id: int,
    enable: list[str],
    disable: list[str],
    use_project_settings: bool | None,
) -> QualitySettings:
    if not enable and not disable and use_project_settings is None:
        raise InvalidArgument("nothing to change: pass enable, disable or use_project_settings")
    if both := sorted(set(enable) & set(disable)):
        raise InvalidArgument(f"in both enable and disable: {', '.join(both)}")
    task = client.tasks.retrieve(task_id)
    if use_project_settings and task.project_id is None:
        raise InvalidArgument(f"task {task_id} is not in a project; it uses its own settings")

    api = client.api_client.quality_api
    own, rows = _load(api, task)
    inherit = own.inherit if use_project_settings is None else use_project_settings
    target, shared = _target(rows, own, inherit)

    # Check every name before any write, so a refusal changes nothing.
    wanted = {name: True for name in enable} | {name: False for name in disable}
    names = [r.name for r in target.requirements]
    for name in wanted:
        if name not in names:
            valid = ", ".join(repr(n) for n in names)
            raise InvalidArgument(f"unknown quality requirement {name!r}; valid: {valid}")
        if names.count(name) > 1:
            raise InvalidArgument(
                f"ambiguous quality requirement {name!r}: {names.count(name)} share it"
            )

    if task.project_id is not None and inherit != own.inherit:
        api.partial_update_settings(
            own.id,
            patched_quality_settings_request=models.PatchedQualitySettingsRequest(inherit=inherit),
        )
    by_name = {r.name: r for r in target.requirements}
    for name, enabled in wanted.items():
        if by_name[name].enabled != enabled:
            api.partial_update_settings_requirements(
                by_name[name].id,
                patched_quality_requirement_request=models.PatchedQualityRequirementRequest(
                    enabled=enabled
                ),
            )
    return _result(task, target, shared, wanted)


def _load(api, task):
    """The task's own settings, plus the project listing for a task in a project."""
    if task.project_id is None:
        return _task_settings(api, task.id), None
    # One listing holds the project's settings and each of its tasks' own settings.
    # shortcut: one page of 500 rows; bigger projects undercount shared_with_tasks
    page, _ = api.list_settings(project_id=task.project_id, page_size=500)
    own = next((s for s in page.results if s.task_id == task.id), None)
    return own or _task_settings(api, task.id), page.results


def _task_settings(api, task_id: int):
    page, _ = api.list_settings(task_id=task_id)
    return page.results[0]


def _target(rows, own, inherit: bool):
    """The settings that decide the task's quality, and how many tasks share them."""
    if rows is None or not inherit:  # a task outside a project says inherit too
        return own, None
    shared = sum(
        1
        for s in rows
        if s.task_id is not None and (inherit if s.task_id == own.task_id else s.inherit)
    )
    return next(s for s in rows if s.task_id is None), shared


def _result(task, target, shared: int | None, wanted: dict[str, bool]) -> QualitySettings:
    return QualitySettings(
        task_id=task.id,
        source="task" if shared is None else "project",
        project_id=task.project_id,
        requirements=[
            Requirement(
                name=r.name,
                annotation_type=str(r.annotation_type),
                enabled=wanted.get(r.name, r.enabled),
            )
            for r in target.requirements
        ],
        shared_with_tasks=shared,
    )
