FROM python:3.9

WORKDIR /app/

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# fix opencv libgl errors
RUN apt-get update && apt-get install libgl1 -y

COPY . /app/
ENV PYTHONPATH=/app