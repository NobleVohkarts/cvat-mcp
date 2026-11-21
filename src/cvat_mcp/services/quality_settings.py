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
    api = client.api_client.quality_api
    own = _task_settings(api, task.id)
    target, listing = _target(api, task, own, own.inherit)
    return _result(task, target, listing, own.inherit, {})


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
    own = _task_settings(api, task_id)
    inherit = own.inherit if use_project_settings is None else use_project_settings
    target, listing = _target(api, task, own, inherit)

    # Check every name before any write, so a refusal changes nothing.
    wanted = {name: True for name in enable} | {name: False for name in disable}
    by_name: dict[str, list] = {}
    for requirement in target.requirements:
        by_name.setdefault(requirement.name, []).append(requirement)
    for name in wanted:
        if name not in by_name:
            valid = ", ".join(f"'{n}'" for n in by_name)
            raise InvalidArgument(f"unknown quality requirement '{name}'; valid: {valid}")
        if len(by_name[name]) > 1:
            raise InvalidArgument(
                f"ambiguous quality requirement '{name}': {len(by_name[name])} share it"
            )

    if task.project_id is not None and inherit != own.inherit:
        api.partial_update_settings(
            own.id,
            patched_quality_settings_request=models.PatchedQualitySettingsRequest(inherit=inherit),
        )
    for name, enabled in wanted.items():
        requirement = by_name[name][0]
        if requirement.enabled != enabled:
            api.partial_update_settings_requirements(
                requirement.id,
                patched_quality_requirement_request=models.PatchedQualityRequirementRequest(
                    enabled=enabled
                ),
            )
    return _result(task, target, listing, inherit, wanted)


def _task_settings(api, task_id: int):
    page, _ = api.list_settings(task_id=task_id)
    return page.results[0]


def _target(api, task, own, inherit: bool):
    """The settings that decide the task's quality, and the project listing when they're its."""
    if task.project_id is None or not inherit:  # a task outside a project says inherit too
        return own, None
    # The project listing holds the project's settings and each of its tasks' own settings.
    listing, _ = api.list_settings(project_id=task.project_id, page_size=500)
    return next(s for s in listing.results if s.task_id is None), listing.results


def _result(task, target, listing, inherit: bool, wanted: dict[str, bool]) -> QualitySettings:
    shared = None
    if listing is not None:
        shared = sum(
            1
            for s in listing
            if s.task_id is not None and (inherit if s.task_id == task.id else s.inherit)
        )
    return QualitySettings(
        task_id=task.id,
        source="task" if listing is None else "project",
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
