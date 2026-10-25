import json
import os
import time

import pandas as pd
import pytest

import algorithm
from algorithm import AlgorithmMemoryError, add_outlying_elements_to_attribute, read_data
from main import DATA_DIR, RESULTS_DIR


def test_rare_value_in_binary_column_is_reported():
    result = add_outlying_elements_to_attribute("flag", pd.Series(["a"] * 100 + ["b"], dtype=str))
    outliers = result["flag"]["outliers"]

    assert outliers, "a 100:1 binary column must report the rare value"
    reported_values = {
        value
        for representatives in outliers.values()
        for counts in representatives.values()
        for value in counts
    }
    assert "b" in reported_values
    assert "a" not in reported_values


def test_balanced_binary_column_has_no_frequency_outlier():
    result = add_outlying_elements_to_attribute("flag", pd.Series(["a"] * 50 + ["b"] * 50, dtype=str))
    assert result["flag"]["outliers"] == {}


def test_rare_value_in_three_value_column_is_reported():
    result = add_outlying_elements_to_attribute(
        "flag", pd.Series(["a"] * 100 + ["b"] * 2 + ["c"], dtype=str)
    )
    reported_values = {
        value
        for representatives in result["flag"]["outliers"].values()
        for counts in representatives.values()
        for value in counts
    }
    assert "c" in reported_values


def test_single_value_column_has_no_outliers():
    result = add_outlying_elements_to_attribute("flag", pd.Series(["x"] * 20, dtype=str))
    assert result["flag"]["outliers"] == {}


def test_latin1_tsv_is_decoded(tmp_path):
    path = tmp_path / "latin1.tsv"
    path.write_bytes("name\nCafé\n".encode("latin-1"))

    assert read_data(str(path))["name"].tolist() == ["Café"]


def test_na_literals_are_preserved(tmp_path):
    path = tmp_path / "na.csv"
    path.write_text("state\nNA\nN/A\nnull\nNone\nNaN\n", encoding="utf-8")

    assert read_data(str(path))["state"].tolist() == ["NA", "N/A", "null", "None", "NaN"]


def test_numeric_json_is_read_as_strings(tmp_path):
    path = tmp_path / "numeric.json"
    path.write_text(json.dumps([{"age": 30}, {"age": 45}, {"age": 30}]), encoding="utf-8")

    frame = read_data(str(path))
    assert frame["age"].tolist() == ["30", "45", "30"]


def test_exactly_uuid_length_values_are_stopped_by_long_column_guard():
    values = ["a" * 29 + f"{index:07d}" for index in range(100)]
    assert {len(value) for value in values} == {36}

    tree = algorithm.tree_grow(algorithm.process_data(pd.Series(values, dtype=str)))
    assert len(tree.leaves) < len(values)


def test_header_only_upload_is_rejected(client, tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("name\n", encoding="utf-8")

    with path.open("rb") as file_handle:
        response = client.post(
            "/api/upload",
            data={"file": (file_handle, "empty.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 400
    assert "no data rows" in response.get_json()["error"].lower()


def test_uploaded_dataset_is_deleted_after_processing(client, tmp_path):
    path = tmp_path / "small.csv"
    path.write_text("flag\na\na\na\nb\n", encoding="utf-8")

    with path.open("rb") as file_handle:
        response = client.post(
            "/api/upload",
            data={"file": (file_handle, "small.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 201
    result_name = response.get_json()["result"]
    job_id = result_name.removesuffix(".json")
    result_path = RESULTS_DIR / result_name
    try:
        assert not (DATA_DIR / f"{job_id}.csv").exists()
        fetched = client.get(f"/api/fetch/{result_name}")
        assert fetched.status_code == 200
    finally:
        result_path.unlink(missing_ok=True)


def test_failed_column_is_reported_instead_of_dropped(monkeypatch, tmp_path):
    path = DATA_DIR / "audit_failure.csv"
    path.write_text("a,b\nx,1\ny,2\n", encoding="utf-8")

    def failed_column(*args, **kwargs):
        return {}, args[0]

    def failed_retry(*args, **kwargs):
        raise MemoryError("simulated")

    monkeypatch.setattr(algorithm, "process_column", failed_column)
    monkeypatch.setattr(algorithm, "add_outlying_elements_to_attribute", failed_retry)
    try:
        with pytest.raises(AlgorithmMemoryError):
            algorithm.process("audit_failure.csv", manual_override_ndistinct=2, first_time=True)
    finally:
        path.unlink(missing_ok=True)
        algorithm.reset_global_values()


def test_expired_results_are_pruned(client, monkeypatch):
    old_result = RESULTS_DIR / "audit_expired.json"
    old_result.write_text("{}", encoding="utf-8")
    old_time = time.time() - 10 * 24 * 60 * 60
    os.utime(old_result, (old_time, old_time))
    monkeypatch.setenv("FIONA_RESULT_RETENTION_DAYS", "7")

    try:
        response = client.get("/api/history")
        assert response.status_code == 200
        assert not old_result.exists()
    finally:
        old_result.unlink(missing_ok=True)
