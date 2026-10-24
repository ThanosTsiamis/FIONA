# Base Image
FROM python:3.11-slim

# set working directory
WORKDIR /usr/src/app

# add and install requirements
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# add app
COPY . /usr/src/app

# run server
CMD python main.py
