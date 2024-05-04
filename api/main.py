import json
import os
from datetime import datetime

import ujson as ujson
from cachetools import cached, TTLCache
from flask import Flask, request, jsonify, redirect, Response

from algorithm import process, logger

app = Flask("FIONA")


@app.errorhandler(500)
def internal_server_error():
    redirection = redirect("http://localhost:3000/error")
    redirection.headers.add('Access-Control-Allow-Origin', '*')
    return redirection


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("resources/data_repository/" + f.filename)
        ndistinct_manual_set = request.form.get('number')
        long_column_cutoff = request.form.get('long_column_cutoff')
        large_file_threshold_input = request.form.get('largeFile_threshold_input')
        if ndistinct_manual_set is not None and ndistinct_manual_set != "":
            ndistinct_manual_set = int(ndistinct_manual_set)
        else:
            # Handle the case when no number is provided
            ndistinct_manual_set = None

        if long_column_cutoff is not None and long_column_cutoff != "":
            long_column_cutoff = int(long_column_cutoff)
        else:
            # Handle the case when no number is provided
            long_column_cutoff = None

        if large_file_threshold_input is not None and large_file_threshold_input != "":
            large_file_threshold_input = int(large_file_threshold_input)
        else:
            # Handle the case when no number is provided
            large_file_threshold_input = None

        outlying_elements = process(f, ndistinct_manual_set, first_time=True,
                                    manual_override_long_column=long_column_cutoff,
                                    manual_override_large_file_threshold=large_file_threshold_input)
        if isinstance(outlying_elements, list):
            columns = outlying_elements
            for column in columns:
                outlying_elements = process(f, ndistinct_manual_set, first_time=False, column_name=column)
                json_serialised = json.dumps(outlying_elements, indent=4)  # Specify indentation level
                basepath = f"resources/json_dumps/multipart_file/{f.filename}/"
                directory = os.path.dirname(basepath)
                os.makedirs(directory, exist_ok=True)
                with open(basepath + column + ".json", "w") as outfile:
                    outfile.write(json_serialised)

            # merge and delete here
            folder_path = "resources/json_dumps/multipart_file"
            output_dir = os.path.dirname(folder_path)

            for root, dirs, files in os.walk(folder_path):
                if root != folder_path:
                    merged_data = {}
                    for filename in files:
                        if filename.endswith(".json"):
                            file_path = os.path.join(root, filename)
                            with open(file_path, "r") as file:
                                json_data = json.load(file)
                                merged_data.update(json_data)

                    subdir_name = os.path.basename(root)
                    merged_json = json.dumps(merged_data, indent=4)  # Specify indentation level

                    output_file = os.path.join(output_dir, subdir_name + ".json")
                    with open(output_file, "w") as file:
                        file.write(merged_json)

                    # Remove the subdirectory
                    # shutil.rmtree(root)
        else:
            json_serialised = json.dumps(outlying_elements, indent=4)  # Specify indentation level
            with open("resources/json_dumps/" + f.filename + ".json", "w") as outfile:
                outfile.write(json_serialised)
        redirection = redirect("http://localhost:3000/results")
        redirection.headers.add('Access-Control-Allow-Origin', '*')
        return redirection


@cached(cache=TTLCache(maxsize=1, ttl=60))
@app.route("/api/fetch/<string:filename>", methods=['GET'])
def fetch(filename):
    file_path = os.path.join("resources/json_dumps", filename)

    if not filename.endswith(".json"):
        file_path += ".json"

    with open(file_path) as f:
        data = ujson.load(f)

    response = Response(ujson.dumps(data), content_type='application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    logger.debug("Response sent")
    return response


@cached(cache=TTLCache(maxsize=1, ttl=180))
@app.route("/api/history", methods=['GET'])
def get_json_files():
    json_files = []
    for file in os.listdir('resources/json_dumps'):
        if file.endswith('.json'):
            json_files.append(file)
    response = jsonify(json_files)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.debug(f'The current time is: {current_time}')
    app.run(host="0.0.0.0", debug=False)
