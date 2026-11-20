from cvat_mcp.services.tasks import clamp_limit, count_objects_by_label


def test_clamp_limit():
    assert clamp_limit(0) == 1
    assert clamp_limit(-5) == 1
    assert clamp_limit(1000) == 100
    assert clamp_limit(20) == 20


def test_count_objects_by_label_empty_keeps_all_labels():
    assert count_objects_by_label([], {1: "car", 2: "bus"}) == {"car": 0, "bus": 0}


def test_count_objects_by_label():
    assert count_objects_by_label([1, 1, 2], {1: "car", 2: "bus"}) == {"car": 2, "bus": 1}
