from dataclasses import dataclass
from datetime import datetime

from . import storage


class UploadValidationError(ValueError):
    pass


@dataclass(frozen=True)
class UploadOptions:
    ndistinct_manual_set: int | None = None
    long_column_cutoff: int | None = None
    large_file_threshold_input: int | None = None
    regex_only: bool = False
    generalised_only: bool = False


def parse_optional_int(value, field_name):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise UploadValidationError(f"Invalid value for '{field_name}': expected an integer.") from exc


def parse_optional_bool(value):
    return value is not None and value != "" and value.lower() == "true"


def parse_upload_options(form):
    return UploadOptions(
        ndistinct_manual_set=parse_optional_int(form.get("number"), "number"),
        long_column_cutoff=parse_optional_int(form.get("long_column_cutoff"), "long_column_cutoff"),
        large_file_threshold_input=parse_optional_int(
            form.get("largeFile_threshold_input"),
            "largeFile_threshold_input",
        ),
        regex_only=parse_optional_bool(form.get("regex_transformation_only")),
        generalised_only=parse_optional_bool(form.get("generalised_transformation_only")),
    )


def process_upload(file_storage, form, *, processor, logger, data_repository_dir, json_dumps_dir):
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise UploadValidationError("No file was uploaded.")

    options = parse_upload_options(form)
    safe_filename = storage.save_uploaded_file(file_storage, data_repository_dir)
    file_storage.filename = safe_filename

    start_time = datetime.now()
    outlying_elements = processor(
        file_storage,
        options.ndistinct_manual_set,
        first_time=True,
        manual_override_long_column=options.long_column_cutoff,
        manual_override_large_file_threshold=options.large_file_threshold_input,
        regex_transformation_only=options.regex_only,
        generalised_transformation_only=options.generalised_only,
    )

    if isinstance(outlying_elements, list):
        for column in outlying_elements:
            column_output = processor(
                file_storage,
                options.ndistinct_manual_set,
                first_time=False,
                column_name=column,
            )
            storage.write_multipart_column_output(
                file_storage.filename,
                column,
                column_output,
                json_dumps_dir,
            )
        storage.merge_multipart_outputs(json_dumps_dir)
    else:
        storage.write_json_output(file_storage.filename, outlying_elements, json_dumps_dir)

    logger.debug(f"Time taken to process the file: {datetime.now() - start_time}")
    return safe_filename
