from pathlib import Path

from main import DATA_DIR, RESULTS_DIR


def test_upload_then_fetch_flights_dirty(client, flights_dirty_csv):
    with flights_dirty_csv.open("rb") as dataset:
        upload = client.post(
            "/api/upload",
            data={"file": (dataset, "flightsDirty.csv")},
            content_type="multipart/form-data",
        )
    assert upload.status_code == 201, upload.get_data(as_text=True)
    body = upload.get_json()
    result_name = body["result"]
    stored_json = RESULTS_DIR / result_name

    try:
        assert result_name.endswith(".json")
        job_id = Path(result_name).stem

        # main.py deletes the uploaded dataset as soon as processing finishes.
        assert not (DATA_DIR / f"{job_id}.csv").exists()
        fetch = client.get(f"/api/fetch/{result_name}")
        assert fetch.status_code == 200, fetch.get_data(as_text=True)
        results = fetch.get_json()
        assert "src" in results
        assert "flight" in results
        for column in ("src", "flight"):
            assert "outliers" in results[column]
            assert "patterns" in results[column]
            assert isinstance(results[column]["outliers"], dict)
            assert isinstance(results[column]["patterns"], dict)
    finally:
        stored_json.unlink(missing_ok=True)
