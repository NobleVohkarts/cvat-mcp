from collections.abc import Iterable
from typing import Literal

from cvat_sdk.api_client import models
from cvat_sdk.attributes import attribute_value_validator
from pydantic import BaseModel, ConfigDict, Field

from cvat_mcp.errors import InvalidArgument

DEFAULT_VALUES = {"checkbox": ["false"], "text": [""]}


class AttributeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_type: Literal["select", "radio", "checkbox", "number", "text"]
    values: list[str] = Field(
        [],
        description="select/radio: the allowed values; number: [min, max, step]; "
        "checkbox: [default]",
    )
    mutable: bool = Field(False, description="may change between frames of a track")


class LabelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["any", "rectangle", "polygon", "polyline", "points", "tag"] = Field(
        "any", description="the kind of object the label may be used for"
    )
    attributes: list[AttributeSpec] = []


def _attribute_request(spec: AttributeSpec) -> models.AttributeRequest:
    request = models.AttributeRequest(
        name=spec.name,
        input_type=spec.input_type,
        values=spec.values or DEFAULT_VALUES.get(spec.input_type, []),
        mutable=spec.mutable,
    )
    try:
        valid = attribute_value_validator(request)  # raises for bad number/select specs
    except ValueError as exc:
        raise InvalidArgument(f"attribute {spec.name!r}: {exc}") from exc
    if spec.input_type == "checkbox" and not valid(request.values[0]):
        raise InvalidArgument(f"attribute {spec.name!r}: a checkbox default is true or false")
    return request


def label_requests(
    specs: list[LabelSpec], existing: Iterable[str] = ()
) -> list[models.PatchedLabelRequest]:
    """CVAT label requests for new labels; names must be new to the task and unique."""
    if not specs:
        raise InvalidArgument("pass at least one label")
    seen = set(existing)
    requests = []
    for spec in specs:
        if not spec.name.strip():
            raise InvalidArgument("a label name must not be empty")
        if spec.name in seen:
            already = spec.name in set(existing)
            raise InvalidArgument(
                f"the task already has a label {spec.name!r}"
                if already
                else f"label {spec.name!r} is listed twice"
            )
        seen.add(spec.name)
        requests.append(
            models.PatchedLabelRequest(
                name=spec.name,
                type=spec.type,
                attributes=[_attribute_request(a) for a in spec.attributes],
            )
        )
    return requests
