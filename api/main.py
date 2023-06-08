import json
import os

from flask import Flask, request, jsonify, redirect, Response

from algorithm import process

app = Flask("FIONA")


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("resources/data_repository/" + f.filename)
        outlying_elements = process(f)
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
    app.run(host="0.0.0.0", debug=True)
    filename = "/json_dumps/dirty.csv"
    big_dict = process(filename)
    json_serialised = json.dumps(big_dict)

