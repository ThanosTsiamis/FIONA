# Base Image
FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# set working directory
WORKDIR /usr/src/app

# add and install requirements first so the layer is cached
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user. The runtime data directories are excluded from the
# build context (see .dockerignore), so create and own them here.
RUN groupadd --system --gid 10001 fiona \
    && useradd --system --uid 10001 --gid fiona --no-create-home --home-dir /usr/src/app fiona

# add app
COPY --chown=fiona:fiona . /usr/src/app
RUN mkdir -p /usr/src/app/resources/data_repository /usr/src/app/resources/json_dumps \
    && chown -R fiona:fiona /usr/src/app

USER fiona

EXPOSE 5000

# One worker: algorithm.py keeps per-run settings in module globals and main.py
# serialises processing with a process-local lock. Threads let fetch/history
# requests be served while a long upload is running.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "1800", "main:app"]
