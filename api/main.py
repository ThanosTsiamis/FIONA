import json
import os
import shutil
from datetime import datetime

from flask import Flask, request, jsonify, redirect, Response

from algorithm import process, logger

app = Flask("FIONA")


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("resources/data_repository/" + f.filename)
        ndistinct_manual_set = request.form.get('number')
        if ndistinct_manual_set is not None and ndistinct_manual_set != "":
            ndistinct_manual_set = int(ndistinct_manual_set)
        else:
            # Handle the case when no number is provided
            ndistinct_manual_set = None

        outlying_elements = process(f, ndistinct_manual_set, first_time=True)
        if isinstance(outlying_elements, list):
            columns = outlying_elements
            for column in columns:
                outlying_elements = process(f, ndistinct_manual_set, first_time=False, column_name=column)
                json_serialised = json.dumps(outlying_elements)
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
                    merged_json = json.dumps(merged_data)

                    output_file = os.path.join(output_dir, subdir_name + ".json")
                    with open(output_file, "w") as file:
                        file.write(merged_json)

                    # Remove the subdirectory
                    shutil.rmtree(root)
        else:
            json_serialised = json.dumps(outlying_elements)
            with open("resources/json_dumps/" + f.filename + ".json", "w") as outfile:
                outfile.write(json_serialised)
        redirection = redirect("http://localhost:3000/results")
        redirection.headers.add('Access-Control-Allow-Origin', '*')
        return redirection


@app.route("/api/fetch/<string:filename>", methods=['GET'])
def fetch(filename):
    if filename.endswith(".json"):
        filepath = "resources/json_dumps/" + filename
    else:
        filepath = "resources/json_dumps/" + filename + ".json"

    def generate_json():
        with open(filepath) as f:
            for line in f:
                yield line.strip()

    response = Response(generate_json(), content_type='application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


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
    app.run(host="0.0.0.0", debug=True)
    filename = "/json_dumps/dirty.csv"
    big_dict = process(filename)
    json_serialised = json.dumps(big_dict)
