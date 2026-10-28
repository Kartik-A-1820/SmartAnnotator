from smartannotator.project import Project


def ann(class_id):
    return {"class_id": class_id, "polygons": [[[0, 0], [4, 0], [4, 4]]], "bbox": [0, 0, 4, 4]}


def test_undo_redo_and_persistence(tmp_path):
    images = [str(tmp_path / f"{i}.png") for i in range(2)]
    project = Project.open(images)
    assert project.path == str(tmp_path / "smartannotator_project.json")
    project.add_annotation(images[0], ann(0))
    project.add_annotation(images[1], ann(0))
    assert project.undo() == images[1]
    assert project.get(images[1]) == []
    assert project.redo() == images[1]
    car = project.add_class("car")
    project.replace_annotation(images[1], 0, dict(ann(0), class_id=car))
    project.save()

    reopened = Project.open(images)
    assert [c["name"] for c in reopened.classes] == ["object", "car"]
    assert reopened.get(images[1])[0]["class_id"] == car
    assert reopened.class_counts() == [1, 1]


def test_nothing_to_undo():
    assert Project().undo() is False and Project().redo() is False


def test_delete_class_shifts_ids_and_is_undoable(tmp_path):
    image = str(tmp_path / "a.png")
    project = Project.open([image])
    project.add_class("b")
    project.add_class("c")
    project.set_annotations(image, [ann(0), ann(1), ann(2)])
    project.delete_class(1)
    assert [a["class_id"] for a in project.get(image)] == [0, 1]
    assert [c["name"] for c in project.classes] == ["object", "c"]
    assert project.undo() is None  # project-wide edit
    assert [a["class_id"] for a in project.get(image)] == [0, 1, 2]
    assert len(project.classes) == 3
