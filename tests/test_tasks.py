from fakes import Obj, fake_client, fake_task

from cvat_mcp.services.tasks import clamp_limit, count_objects_by_label, get_task_overview


def test_clamp_limit():
    assert clamp_limit(0) == 1
    assert clamp_limit(-5) == 1
    assert clamp_limit(1000) == 100
    assert clamp_limit(20) == 20


def test_count_objects_by_label_empty_keeps_all_labels():
    assert count_objects_by_label([], {1: "car", 2: "bus"}) == {"car": 0, "bus": 0}


def test_count_objects_by_label():
    assert count_objects_by_label([1, 1, 2], {1: "car", 2: "bus"}) == {"car": 2, "bus": 1}


class _NoProjectAccess:
    def retrieve(self, _id):
        raise AssertionError("must not request the project: the user may lack access")


def test_overview_uses_task_project_name_and_counts_tags():
    one = Obj(label_id=1)
    task = fake_task(
        [one], tracks=[one], tags=[one], name="t", project_id=2, project_name="Traffic", task_id=3
    )
    overview = get_task_overview(fake_client(task, projects=_NoProjectAccess()), 3)
    assert overview.project_name == "Traffic"
    assert overview.objects_per_label == {"car": 3, "person": 0}
    assert overview.total_objects == 3
