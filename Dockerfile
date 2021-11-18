FROM eggroll/uvicorn-dev:python3.9

WORKDIR /app/
RUN apt-get update && apt-get install libgl1 -y

RUN pip install openai==0.10.5

COPY . /app/
ENV PYTHONPATH=/app