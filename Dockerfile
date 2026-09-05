FROM python:3.14-slim

# Install minimal system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

COPY config/ /app/config/

COPY input/ /app/input/

COPY model/ /app/model/

RUN mkdir /app/output

WORKDIR /app/src

CMD ["python", "main.py"]
