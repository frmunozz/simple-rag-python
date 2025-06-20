from python:3.10-slim

RUN apt-get update && apt-get install -y python3-dev libsqlite3-0 curl gcc

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

