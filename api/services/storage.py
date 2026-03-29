import json
import os

import ujson


def sanitize_filename(filename):
    return os.path.basename(filename)


def save_uploaded_file(file_storage, data_repository_dir):
    filename = sanitize_filename(file_storage.filename)
    os.makedirs(data_repository_dir, exist_ok=True)
    destination = os.path.join(data_repository_dir, filename)
    file_storage.save(destination)
    return filename


def write_json_output(filename, payload, json_dumps_dir):
    os.makedirs(json_dumps_dir, exist_ok=True)
    output_path = os.path.join(json_dumps_dir, f"{filename}.json")
    with open(output_path, "w") as outfile:
        outfile.write(json.dumps(payload, indent=4))


def write_multipart_column_output(filename, column_name, payload, json_dumps_dir):
    basepath = os.path.join(json_dumps_dir, "multipart_file", filename)
    os.makedirs(basepath, exist_ok=True)
    output_path = os.path.join(basepath, f"{column_name}.json")
    with open(output_path, "w") as outfile:
        outfile.write(json.dumps(payload, indent=4))


def merge_multipart_outputs(json_dumps_dir):
    folder_path = os.path.join(json_dumps_dir, "multipart_file")
    output_dir = os.path.dirname(folder_path)

    for root, _, files in os.walk(folder_path):
        if root == folder_path:
            continue

        merged_data = {}
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                with open(file_path, "r") as file:
                    merged_data.update(json.load(file))

        subdir_name = os.path.basename(root)
        output_file = os.path.join(output_dir, subdir_name + ".json")
        with open(output_file, "w") as file:
            file.write(json.dumps(merged_data, indent=4))


def read_json_output(filename, json_dumps_dir):
    file_path = os.path.join(json_dumps_dir, filename)
    if not filename.endswith(".json"):
        file_path += ".json"

    with open(file_path) as file:
        return ujson.load(file)


def list_json_files(json_dumps_dir):
    if not os.path.isdir(json_dumps_dir):
        return []

    return [file for file in os.listdir(json_dumps_dir) if file.endswith(".json")]
