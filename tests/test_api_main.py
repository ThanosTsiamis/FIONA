import json
import os
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest import mock

from tests.module_loader import load_module


algorithm_stub = types.SimpleNamespace(
    process=lambda *args, **kwargs: {},
    logger=types.SimpleNamespace(debug=lambda *args, **kwargs: None),
)

main = load_module(
    "test_main_module",
    "api/main.py",
    extra_modules={"algorithm": algorithm_stub},
)


class MainRoutesTest(unittest.TestCase):
    @contextmanager
    def in_temp_app_dir(self, processor=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_repository_dir = os.path.join(temp_dir, "resources", "data_repository")
            json_dumps_dir = os.path.join(temp_dir, "resources", "json_dumps")
            os.makedirs(data_repository_dir)
            os.makedirs(json_dumps_dir)

            previous_cwd = os.getcwd()
            previous_config = dict(getattr(main.current_app, "config", {}))
            try:
                os.chdir(temp_dir)
                main.current_app.config = {
                    "DATA_REPOSITORY_DIR": data_repository_dir,
                    "JSON_DUMPS_DIR": json_dumps_dir,
                    "FRONTEND_BASE_URL": "http://localhost:3000",
                    "PROCESSOR": processor or algorithm_stub.process,
                    "LOGGER": algorithm_stub.logger,
                }
                yield temp_dir
            finally:
                main.current_app.config = previous_config
                os.chdir(previous_cwd)

    def test_create_app_sets_default_dependencies(self):
        app = main.create_app()

        self.assertEqual(app.config["FRONTEND_BASE_URL"], "http://localhost:3000")
        self.assertIs(app.config["PROCESSOR"], main.process)
        self.assertIs(app.config["LOGGER"], main.logger)

    def test_internal_server_error_redirects_to_error_page(self):
        with self.in_temp_app_dir():
            response = main.internal_server_error(Exception("boom"))

        self.assertEqual(response.location, "http://localhost:3000/error")
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_fetch_reads_json_file_and_sets_cors_header(self):
        with self.in_temp_app_dir():
            file_path = os.path.join(main.current_app.config["JSON_DUMPS_DIR"], "sample.json")
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump({"status": "ok"}, handle)

            response = main.fetch("sample")

        self.assertEqual(response.content_type, "application/json")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.response), {"status": "ok"})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_health_returns_ok_payload(self):
        with self.in_temp_app_dir():
            response = main.health()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.response), {"status": "ok"})
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_get_json_files_returns_only_json_entries(self):
        with self.in_temp_app_dir():
            for filename in ("a.json", "b.json", "notes.txt"):
                with open(
                    os.path.join(main.current_app.config["JSON_DUMPS_DIR"], filename),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write("{}")

            response = main.get_json_files()

        self.assertEqual(sorted(response.response), ["a.json", "b.json"])
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_upload_file_writes_single_json_result_and_redirects(self):
        class FakeFile:
            filename = "dataset.csv"

            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("id,value\n1,test\n")

        processor = mock.Mock(return_value={"email": {"patterns": {"90.0": {"U+l": {"A1@example.com": 1}}}}})
        request_stub = types.SimpleNamespace(
            method="POST",
            files={"file": FakeFile()},
            form={
                "number": "4",
                "long_column_cutoff": "15",
                "largeFile_threshold_input": "50",
                "regex_transformation_only": "true",
                "generalised_transformation_only": "",
            },
        )

        with self.in_temp_app_dir(processor=processor):
            with mock.patch.object(main, "request", request_stub):
                response = main.upload_file()

            output_path = os.path.join(main.current_app.config["JSON_DUMPS_DIR"], "dataset.csv.json")
            with open(output_path, "r", encoding="utf-8") as handle:
                written_payload = json.load(handle)

        self.assertEqual(response.location, "http://localhost:3000/results")
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
        self.assertEqual(written_payload, {"email": {"patterns": {"90.0": {"U+l": {"A1@example.com": 1}}}}})
        processor.assert_called_once()
        args, kwargs = processor.call_args
        self.assertEqual(args[1], 4)
        self.assertTrue(kwargs["first_time"])
        self.assertEqual(kwargs["manual_override_long_column"], 15)
        self.assertEqual(kwargs["manual_override_large_file_threshold"], 50)
        self.assertTrue(kwargs["regex_transformation_only"])
        self.assertFalse(kwargs["generalised_transformation_only"])

    def test_upload_file_merges_multipart_results(self):
        class FakeFile:
            filename = "multipart.csv"

            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("a,b\nx,y\n")

        processor = mock.Mock(
            side_effect=[
                ["column_a", "column_b"],
                {"column_a": {"outliers": {"10": {"U+": {"AA": 1}}}}},
                {"column_b": {"outliers": {"20": {"d+": {"99": 1}}}}},
            ]
        )
        request_stub = types.SimpleNamespace(
            method="POST",
            files={"file": FakeFile()},
            form={
                "number": "",
                "long_column_cutoff": "",
                "largeFile_threshold_input": "",
                "regex_transformation_only": "",
                "generalised_transformation_only": "",
            },
        )

        with self.in_temp_app_dir(processor=processor):
            with mock.patch.object(main, "request", request_stub):
                response = main.upload_file()

            merged_path = os.path.join(main.current_app.config["JSON_DUMPS_DIR"], "multipart.csv.json")
            column_a_path = os.path.join(
                main.current_app.config["JSON_DUMPS_DIR"], "multipart_file", "multipart.csv", "column_a.json"
            )
            column_b_path = os.path.join(
                main.current_app.config["JSON_DUMPS_DIR"], "multipart_file", "multipart.csv", "column_b.json"
            )

            with open(merged_path, "r", encoding="utf-8") as handle:
                merged_payload = json.load(handle)

            column_a_exists = os.path.exists(column_a_path)
            column_b_exists = os.path.exists(column_b_path)

        self.assertEqual(response.location, "http://localhost:3000/results")
        self.assertTrue(column_a_exists)
        self.assertTrue(column_b_exists)
        self.assertEqual(
            merged_payload,
            {
                "column_a": {"outliers": {"10": {"U+": {"AA": 1}}}},
                "column_b": {"outliers": {"20": {"d+": {"99": 1}}}},
            },
        )
        self.assertEqual(processor.call_count, 3)
        first_call_args, first_call_kwargs = processor.call_args_list[0]
        self.assertIsNone(first_call_args[1])
        self.assertTrue(first_call_kwargs["first_time"])
        self.assertFalse(first_call_kwargs["regex_transformation_only"])
        self.assertFalse(first_call_kwargs["generalised_transformation_only"])
        self.assertEqual(processor.call_args_list[1].kwargs["column_name"], "column_a")
        self.assertFalse(processor.call_args_list[1].kwargs["first_time"])
        self.assertEqual(processor.call_args_list[2].kwargs["column_name"], "column_b")
        self.assertFalse(processor.call_args_list[2].kwargs["first_time"])

    def test_upload_file_rejects_missing_file(self):
        request_stub = types.SimpleNamespace(method="POST", files={}, form={})
        processor = mock.Mock()

        with self.in_temp_app_dir(processor=processor):
            with mock.patch.object(main, "request", request_stub):
                response = main.upload_file()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.response), {"error": "No file was uploaded."})
        processor.assert_not_called()

    def test_upload_file_rejects_invalid_integer_fields(self):
        class FakeFile:
            filename = "bad.csv"

            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("id\n1\n")

        processor = mock.Mock()
        request_stub = types.SimpleNamespace(
            method="POST",
            files={"file": FakeFile()},
            form={"number": "not-a-number"},
        )

        with self.in_temp_app_dir(processor=processor):
            with mock.patch.object(main, "request", request_stub):
                response = main.upload_file()

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            json.loads(response.response),
            {"error": "Invalid value for 'number': expected an integer."},
        )
        processor.assert_not_called()

    def test_upload_file_propagates_processing_failures(self):
        class FakeFile:
            filename = "broken.csv"

            def save(self, path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("id\n1\n")

        processor = mock.Mock(side_effect=RuntimeError("boom"))
        request_stub = types.SimpleNamespace(method="POST", files={"file": FakeFile()}, form={})

        with self.in_temp_app_dir(processor=processor):
            with mock.patch.object(main, "request", request_stub):
                with self.assertRaises(RuntimeError):
                    main.upload_file()

    def test_fetch_returns_404_json_for_missing_file(self):
        with self.in_temp_app_dir():
            response = main.fetch("missing-file")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(
            json.loads(response.response),
            {"error": "File 'missing-file' was not found."},
        )
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_get_json_files_returns_empty_list_when_directory_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_config = dict(getattr(main.current_app, "config", {}))
            previous_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                main.current_app.config = {
                    "JSON_DUMPS_DIR": os.path.join(temp_dir, "resources", "json_dumps"),
                }
                response = main.get_json_files()
            finally:
                main.current_app.config = previous_config
                os.chdir(previous_cwd)

        self.assertEqual(response.response, [])
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")


if __name__ == "__main__":
    unittest.main()
