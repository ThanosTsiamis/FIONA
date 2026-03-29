import csv
import os
import shutil
import tempfile
import unittest
from pathlib import Path


try:
    from api.algorithm import process
except Exception:  # pragma: no cover - skipped when real backend deps are unavailable
    process = None


DATASET_ROOT = Path(__file__).resolve().parents[1] / "api" / "resources" / "datasets" / "datasets_testing_purposes"


class FakeUpload:
    def __init__(self, filename):
        self.filename = filename


@unittest.skipUnless(process is not None, "Real algorithm dependencies are not installed.")
class AlgorithmRegressionTest(unittest.TestCase):
    def test_dirty_sample_datasets_produce_structured_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_repository_dir = Path(temp_dir) / "resources" / "data_repository"
            data_repository_dir.mkdir(parents=True)
            previous_cwd = os.getcwd()

            try:
                os.chdir(temp_dir)
                for dataset_path in DATASET_ROOT.glob("*/*Dirty.csv"):
                    copied_path = data_repository_dir / dataset_path.name
                    shutil.copy(dataset_path, copied_path)
                    with dataset_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                        headers = next(csv.reader(handle))

                    result = process(
                        FakeUpload(dataset_path.name),
                        manual_override_ndistinct=2,
                        manual_override_large_file_threshold=10_000_000,
                    )

                    self.assertIsInstance(result, dict, dataset_path.name)
                    self.assertTrue(result, dataset_path.name)
                    self.assertTrue(set(result.keys()).issubset(set(headers)), dataset_path.name)
                    for column_payload in result.values():
                        self.assertIn("outliers", column_payload)
                        self.assertIn("patterns", column_payload)
            finally:
                os.chdir(previous_cwd)
