from project import Project


def ann(cid):
    return {"class_id": cid, "polygons": [[[0, 0], [4, 0], [4, 4]]], "bbox": [0, 0, 4, 4]}


def test_undo_redo_and_persistence(tmp_path):
    imgs = [str(tmp_path / f"{i}.png") for i in range(2)]
    p = Project()
    p.open_images(imgs)
    assert p.path == str(tmp_path / "smartannotator_project.json")
    p.add_annotation(imgs[0], ann(0))
    p.add_annotation(imgs[1], ann(0))
    assert p.undo() == (True, imgs[1])
    assert p.get(imgs[1]) == []
    assert p.redo() == (True, imgs[1])
    cid = p.add_class("car")
    p.change_class(imgs[1], 0, cid)
    p.save()

    q = Project()
    q.open_images(imgs)
    assert [c["name"] for c in q.classes] == ["object", "car"]
    assert q.get(imgs[1])[0]["class_id"] == 1
    assert q.class_counts() == [1, 1]


def test_delete_class_shifts_ids_and_undoes(tmp_path):
    img = str(tmp_path / "a.png")
    p = Project()
    p.open_images([img])
    p.add_class("b")
    p.add_class("c")
    p.set_annotations(img, [ann(0), ann(1), ann(2)])
    p.delete_class(1)
    assert [a["class_id"] for a in p.get(img)] == [0, 1]
    assert [c["name"] for c in p.classes] == ["object", "c"]
    p.undo()
    assert [a["class_id"] for a in p.get(img)] == [0, 1, 2]
    assert len(p.classes) == 3
