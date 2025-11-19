FROM python:3.11-slim

RUN apt-get update && apt-get install -y git curl
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ..

CMD ["python", "/app/src/main.py"]
