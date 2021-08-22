FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8


COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt --upgrade

COPY ./app /app/
