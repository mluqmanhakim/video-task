FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

COPY config/ /app/config/

RUN mkdir /data

WORKDIR /app/src
CMD ["python", "main.py"]
