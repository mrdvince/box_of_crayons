FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR /app/

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/
ENV PYTHONPATH=/app