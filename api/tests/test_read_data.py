import json

import pandas as pd
import pytest

from algorithm import read_data


def test_read_data_csv_and_column_subset(flights_dirty_csv):
    full = read_data(str(flights_dirty_csv))
    assert list(full.columns)[:2] == ["tuple_id", "src"]
    assert len(full) > 0

    src_only = read_data(str(flights_dirty_csv), column_name="src")
    assert list(src_only.columns) == ["src"]
    assert len(src_only) == len(full)


def test_read_data_tsv(tmp_path):
    path = tmp_path / "sample.tsv"
    path.write_text("name\tcode\nAlpha\tA1\nBeta\tB2\n", encoding="utf-8")
    frame = read_data(str(path))
    assert list(frame.columns) == ["name", "code"]
    assert frame.iloc[0]["name"] == "Alpha"


def test_read_data_json(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps([{"name": "Alpha", "code": "A1"}, {"name": "Beta", "code": "B2"}]), encoding="utf-8")
    frame = read_data(str(path))
    assert set(frame.columns) == {"name", "code"}
    assert len(frame) == 2


def test_read_data_xlsx(tmp_path):
    path = tmp_path / "sample.xlsx"
    pd.DataFrame({"name": ["Alpha", "Beta"], "code": ["A1", "B2"]}).to_excel(path, index=False)
    frame = read_data(str(path))
    assert list(frame.columns) == ["name", "code"]
    assert frame.iloc[1]["code"] == "B2"


def test_read_data_rejects_unknown_extension(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_data(str(path))
