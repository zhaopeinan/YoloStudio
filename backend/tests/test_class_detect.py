from __future__ import annotations

from pathlib import Path

from app.datasets.class_detect import detect_classes_from_filenames, extract_class_prefix
from test_dataset_import_api import create_import_zip, isolated_client


def test_extract_class_prefix_from_h_a_filenames():
    assert extract_class_prefix("fire_truck_h20_a045.jpg") == "fire_truck"
    assert extract_class_prefix("prius_hybrid_camo_h75_a225.JPG") == "prius_hybrid_camo"
    assert extract_class_prefix("person_red_h80_a045.jpg") == "person_red"
    assert extract_class_prefix("._suv_camo_h20_a045.jpg") is None


def test_detect_classes_from_filenames_groups_prefixes():
    names = [
        "fire_truck_h20_a045.jpg",
        "fire_truck_h30_a090.jpg",
        "person_white_h20_a000.jpg",
        "person_white_h40_a180.jpg",
        "prius_hybrid_h10_a000.jpg",
        "unrelated.jpg",
        "._fire_truck_h20_a045.jpg",
        "._suv_camo_h20_a045.jpg",
        "._suv_camo_h30_a090.jpg",
    ]
    items, method = detect_classes_from_filenames(names, min_count=2)
    assert method == "filename_prefix_h_a"
    by_name = {item["name"]: item["image_count"] for item in items}
    assert by_name == {"fire_truck": 2, "person_white": 2}
    assert "prius_hybrid" not in by_name  # only one image, below min_count
    assert "._suv_camo" not in by_name
    assert "._fire_truck" not in by_name


def test_detected_classes_endpoint_after_import(tmp_path: Path):
    zip_path = tmp_path / "sample.zip"
    # Build a zip with the domain naming convention.
    from zipfile import ZipFile

    with ZipFile(zip_path, "w") as archive:
        for name in [
            "images/g/fire_truck_h20_a045.jpg",
            "images/g/fire_truck_h30_a090.jpg",
            "images/g/person_white_h20_a000.jpg",
            "images/g/person_white_h40_a180.jpg",
            "images/g/car_lexus_h10_a000.jpg",
            "images/g/car_lexus_h20_a045.jpg",
        ]:
            archive.writestr(name, b"fake")

    with isolated_client(tmp_path) as client:
        imported = client.post(
            "/api/datasets/import",
            json={
                "source_path": str(zip_path),
                "project_name": "Detect Classes Project",
                "dataset_name": "sample",
            },
        )
        assert imported.status_code == 200
        dataset_id = imported.json()["dataset_id"]

        response = client.get(f"/api/datasets/{dataset_id}/detected-classes")
        assert response.status_code == 200
        payload = response.json()
        assert payload["total_images"] == 6
        assert payload["method"] == "filename_prefix_h_a"
        names = [item["name"] for item in payload["items"]]
        assert names == ["car_lexus", "fire_truck", "person_white"]
        assert all(item["image_count"] == 2 for item in payload["items"])
        assert all(item["color"].startswith("#") for item in payload["items"])
