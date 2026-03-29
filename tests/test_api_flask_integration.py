import io
import json
import os
import tempfile
import unittest


try:
    from api.main import create_app
except Exception:  # pragma: no cover - skipped when real backend deps are unavailable
    create_app = None


@unittest.skipUnless(create_app is not None, "Real Flask dependencies are not installed.")
class FlaskIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_repository_dir = os.path.join(self.temp_dir.name, "resources", "data_repository")
        self.json_dumps_dir = os.path.join(self.temp_dir.name, "resources", "json_dumps")
        os.makedirs(self.data_repository_dir)
        os.makedirs(self.json_dumps_dir)
        self.processor_calls = []

        def processor(*args, **kwargs):
            self.processor_calls.append((args, kwargs))
            if kwargs.get("first_time", True):
                return {"email": {"outliers": {}, "patterns": {}}}
            return {"unused": {"outliers": {}, "patterns": {}}}

        self.app = create_app(
            {
                "TESTING": True,
                "DATA_REPOSITORY_DIR": self.data_repository_dir,
                "JSON_DUMPS_DIR": self.json_dumps_dir,
            },
            processor=processor,
        )
        self.client = self.app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_upload_endpoint_saves_output_and_redirects(self):
        response = self.client.post(
            "/api/upload",
            data={
                "file": (io.BytesIO(b"id,value\n1,test\n"), "sample.csv"),
                "number": "4",
                "long_column_cutoff": "15",
                "largeFile_threshold_input": "50",
                "regex_transformation_only": "true",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith("/results"))
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

        output_path = os.path.join(self.json_dumps_dir, "sample.csv.json")
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"email": {"outliers": {}, "patterns": {}}})

        _, kwargs = self.processor_calls[0]
        self.assertEqual(kwargs["manual_override_long_column"], 15)
        self.assertEqual(kwargs["manual_override_large_file_threshold"], 50)
        self.assertTrue(kwargs["regex_transformation_only"])

    def test_fetch_endpoint_returns_json_payload(self):
        with open(os.path.join(self.json_dumps_dir, "saved.json"), "w", encoding="utf-8") as handle:
            json.dump({"status": "ok"}, handle)

        response = self.client.get("/api/fetch/saved")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_history_endpoint_lists_json_files(self):
        for filename in ("a.json", "b.json", "ignore.txt"):
            with open(os.path.join(self.json_dumps_dir, filename), "w", encoding="utf-8") as handle:
                handle.write("{}")

        response = self.client.get("/api/history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(sorted(response.get_json()), ["a.json", "b.json"])
