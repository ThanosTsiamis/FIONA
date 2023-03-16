import json
import time

from flask import Flask, request, jsonify, redirect

from algorithm import process

app = Flask("FIONA")


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("../resources/json_dumps/" + f.filename)
        outlying_elements = process(file=f, multiprocess_switch=request.form['enableParallel'])
        json_serialised = json.dumps(outlying_elements)
        with open("../resources/json_dumps/" + f.filename + ".json", "w") as outfile:
            outfile.write(json_serialised)
        redirection = redirect("http://localhost:3000/results")
        redirection.headers.add('Access-Control-Allow-Origin', '*')
        return redirection


@app.route("/api/fetch/<string:filename>", methods=['GET'])
def fetch(filename):
    with open("../resources/json_dumps/" + filename + ".json") as f:
        data = json.load(f)
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=False)
    t = time.time()
    filename = "../json_dumps/testing123.csv"
    multiprocess = False
    big_dict = process(filename, multiprocess)
    json_serialised = json.dumps(big_dict)
    with open("../resources/json_dumps/" + "jhjhbjhbhj.json", "w") as outfile:
        outfile.write(json_serialised)
    print(time.time() - t)
