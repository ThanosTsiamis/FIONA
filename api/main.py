import os
from datetime import datetime

import ujson
from cachetools import TTLCache, cached
from flask import Flask, Response, current_app, jsonify, redirect, request

try:
    from .algorithm import logger, process
    from .services.storage import list_json_files, read_json_output
    from .services.upload_service import UploadValidationError, process_upload
except ImportError:  # pragma: no cover - local fallback for direct execution/tests
    from algorithm import logger, process
    from services.storage import list_json_files, read_json_output
    from services.upload_service import UploadValidationError, process_upload


DEFAULT_CONFIG = {
    "DATA_REPOSITORY_DIR": os.path.join("resources", "data_repository"),
    "JSON_DUMPS_DIR": os.path.join("resources", "json_dumps"),
    "FRONTEND_BASE_URL": "http://localhost:3000",
}


def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


def json_response(payload, status_code):
    response = Response(ujson.dumps(payload), content_type="application/json")
    response.status_code = status_code
    return add_cors_headers(response)


def build_results_redirect():
    return add_cors_headers(redirect(f"{current_app.config['FRONTEND_BASE_URL']}/results"))


def internal_server_error(error):
    return add_cors_headers(redirect(f"{current_app.config['FRONTEND_BASE_URL']}/error"))


def upload_file():
    if request.method != "POST":
        return json_response({"error": "Method not allowed."}, 405)

    file_storage = request.files.get("file")
    try:
        process_upload(
            file_storage,
            request.form,
            processor=current_app.config["PROCESSOR"],
            logger=current_app.config["LOGGER"],
            data_repository_dir=current_app.config["DATA_REPOSITORY_DIR"],
            json_dumps_dir=current_app.config["JSON_DUMPS_DIR"],
        )
    except UploadValidationError as error:
        return json_response({"error": str(error)}, 400)

    return build_results_redirect()


def health():
    return json_response({"status": "ok"}, 200)


@cached(cache=TTLCache(maxsize=1, ttl=60))
def fetch(filename):
    try:
        data = read_json_output(filename, current_app.config["JSON_DUMPS_DIR"])
    except FileNotFoundError:
        return json_response({"error": f"File '{filename}' was not found."}, 404)

    response = Response(ujson.dumps(data), content_type="application/json")
    current_app.config["LOGGER"].debug("Response sent")
    return add_cors_headers(response)


@cached(cache=TTLCache(maxsize=1, ttl=180))
def get_json_files():
    response = jsonify(list_json_files(current_app.config["JSON_DUMPS_DIR"]))
    return add_cors_headers(response)


def create_app(test_config=None, *, processor=process, app_logger=logger):
    app = Flask("FIONA")
    app.config.update(DEFAULT_CONFIG)
    if test_config:
        app.config.update(test_config)
    app.config["PROCESSOR"] = processor
    app.config["LOGGER"] = app_logger

    app.errorhandler(500)(internal_server_error)
    app.route("/api/upload", methods=["POST"])(upload_file)
    app.route("/api/health", methods=["GET"])(health)
    app.route("/api/fetch/<string:filename>", methods=["GET"])(fetch)
    app.route("/api/history", methods=["GET"])(get_json_files)
    return app


app = create_app()


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug(f"The current time is: {current_time}")
    app.run(host="0.0.0.0", debug=False)
