import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from uuid import uuid4

import ujson as ujson
from flask import Flask, Response, jsonify, request

from algorithm import AlgorithmMemoryError, process, logger, reset_global_values, safe_filename

# Logging belongs to the application entry point, not to the algorithm module.
logging.basicConfig(level=getattr(logging, os.environ.get("FIONA_LOG_LEVEL", "INFO").upper(), logging.INFO))

app = Flask("FIONA")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "resources" / "data_repository"
RESULTS_DIR = BASE_DIR / "resources" / "json_dumps"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json", ".tsv"}
DEFAULT_RESULT_RETENTION_DAYS = 7
# algorithm.py currently keeps per-run settings in module globals. Keep that
# state isolated until it can be refactored into an explicit configuration object.
PROCESSING_LOCK = Lock()


def result_retention_days():
    raw = os.environ.get("FIONA_RESULT_RETENTION_DAYS", str(DEFAULT_RESULT_RETENTION_DAYS))
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid FIONA_RESULT_RETENTION_DAYS=%r", raw)
        return float(DEFAULT_RESULT_RETENTION_DAYS)


def prune_old_results():
    """Delete result files older than the configured retention period (0 disables pruning)."""
    retention_days = result_retention_days()
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 24 * 60 * 60
    for path in RESULTS_DIR.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            logger.warning("Could not remove expired result %s", path, exc_info=True)


def prune_old_uploads():
    """Remove datasets left behind by a crashed request (0 disables pruning)."""
    retention_days = result_retention_days()
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 24 * 60 * 60
    for path in DATA_DIR.iterdir():
        if not path.is_file() or path.name == ".gitkeep":
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            logger.warning("Could not remove expired upload %s", path, exc_info=True)


def json_response(payload, status=200):
    response = jsonify(payload)
    response.status_code = status
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def parse_optional_positive_int(value, field_name):
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a whole number.")
    if parsed <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return parsed


def parse_boolean(value):
    return str(value).lower() == "true"


def result_path(filename):
    """Return a result path only when *filename* is a single safe component."""
    safe_name = safe_filename(filename)
    if safe_name != filename:
        return None
    if not safe_name.endswith(".json"):
        safe_name += ".json"
    return RESULTS_DIR / safe_name


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Unhandled server error", exc_info=error)
    return json_response({"error": "The server could not process this request."}, 500)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    f = request.files.get('file')
    if f is None or not f.filename:
        return json_response({"error": "A dataset file is required."}, 400)

    original_name = safe_filename(f.filename)
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return json_response({"error": "Unsupported file format. Use CSV, XLSX, JSON, or TSV."}, 400)

    try:
        ndistinct_manual_set = parse_optional_positive_int(request.form.get('number'), 'number')
        long_column_cutoff = parse_optional_positive_int(request.form.get('long_column_cutoff'), 'long_column_cutoff')
        large_file_threshold_input = parse_optional_positive_int(
            request.form.get('largeFile_threshold_input'), 'largeFile_threshold_input')
    except ValueError as error:
        return json_response({"error": str(error)}, 400)

    # Do not use a client-supplied name as a storage key: concurrent uploads must
    # neither overwrite each other nor expose each other's results.
    job_id = uuid4().hex
    stored_name = f"{job_id}{extension}"
    stored_path = DATA_DIR / stored_name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prune_old_results()
    prune_old_uploads()

    try:
        f.save(stored_path)

        regex_only = parse_boolean(request.form.get('regex_transformation_only'))
        generalised_only = parse_boolean(request.form.get('generalised_transformation_only'))
        with PROCESSING_LOCK:
            start_time = datetime.now()
            try:
                outlying_elements = process(stored_name, ndistinct_manual_set, first_time=True,
                                            manual_override_long_column=long_column_cutoff,
                                            manual_override_large_file_threshold=large_file_threshold_input,
                                            regex_transformation_only=regex_only,
                                            generalised_transformation_only=generalised_only)
                if isinstance(outlying_elements, list):
                    merged_data = {}
                    for column in outlying_elements:
                        column_result = process(stored_name, ndistinct_manual_set, first_time=False, column_name=column,
                                                regex_transformation_only=regex_only,
                                                generalised_transformation_only=generalised_only)
                        merged_data.update(column_result)
                    outlying_elements = merged_data
            except ValueError as error:
                return json_response({"error": str(error)}, 400)
            except AlgorithmMemoryError as error:
                logger.warning("A column could not be processed: %s", error, exc_info=True)
                return json_response({"error": str(error)}, 500)

            result_name = f"{job_id}.json"
            with (RESULTS_DIR / result_name).open("w", encoding="utf-8") as outfile:
                json.dump(outlying_elements, outfile, indent=4)
            end_time = datetime.now()
            time_taken = end_time - start_time
            logger.debug(f"Time taken to process the file: {time_taken}")
            return json_response({"result": result_name}, 201)
    finally:
        reset_global_values()
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove uploaded dataset %s", stored_path, exc_info=True)


@app.route("/api/fetch/<string:filename>", methods=['GET'])
def fetch(filename):
    file_path = result_path(filename)
    if file_path is None or not file_path.is_file():
        return json_response({"error": "Result not found."}, 404)

    with file_path.open(encoding="utf-8") as f:
        data = ujson.load(f)

    response = Response(ujson.dumps(data), content_type='application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    logger.debug("Response sent")
    return response


@app.route("/api/history", methods=['GET'])
def get_json_files():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prune_old_results()
    return json_response(sorted(path.name for path in RESULTS_DIR.glob('*.json')))


if __name__ == "__main__":
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.debug(f'The current time is: {current_time}')
    app.run(host="0.0.0.0", debug=False)
    # outlying_elements = process("datasets_testing_purposes/flights/flightsDirty.csv", first_time=True,
    #                             manual_override_long_column=None,
    #                             manual_override_large_file_threshold=2)
    #
    # if isinstance(outlying_elements, list):
    #     outlying_elements=process("datasets_testing_purposes/flights/flightsDirty.csv", first_time=False, column_name=outlying_elements[0])
