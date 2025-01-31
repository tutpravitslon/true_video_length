FROM python:3.10.14

RUN set -ex \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        wget \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
        libpq-dev \
        libmagic-dev \
        libbz2-dev \
        lzma \
        liblzma-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# COPY ./main.py /app/main.py
# COPY ./digit_classifier.onnx /app/digit_classifier.onnx
# COPY ./image_processing.py /app/image_processing.py
# COPY ./timestamp_parser.py /app/timestamp_parser.py
# COPY ./utils.py /app/utils.py
# COPY ./config.json /app/config.json

WORKDIR /app

COPY . .
ENTRYPOINT ["python3", "main.py"]


