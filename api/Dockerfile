# Base Image
FROM python:3.11-slim

# set working directory
WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# system packages needed for building native Python dependencies such as psutil
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# add and install requirements
COPY api/requirements.txt /usr/src/app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# add app
COPY api /usr/src/app/api

# document the Flask port
EXPOSE 5000

# run server
CMD ["python", "-m", "api.main"]
