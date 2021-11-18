FROM python:3.9.9-slim

WORKDIR /app/

COPY ./requirements.txt /tmp/requirements.txt
COPY ./scripts/start.sh /start.sh
COPY ./scripts/start-reload.sh /start-reload.sh

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# fix opencv libgl 
RUN apt-get update && apt-get install libgl1 -y

COPY . /app/
ENV PYTHONPATH=/app