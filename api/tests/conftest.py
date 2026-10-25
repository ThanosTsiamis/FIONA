from pathlib import Path

import pytest

from main import app

FLIGHTS_DIRTY = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "datasets"
    / "datasets_testing_purposes"
    / "flights"
    / "flightsDirty.csv"
)


@pytest.fixture
def client():
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture
def flights_dirty_csv():
    assert FLIGHTS_DIRTY.is_file(), f"Missing fixture dataset: {FLIGHTS_DIRTY}"
    return FLIGHTS_DIRTY
