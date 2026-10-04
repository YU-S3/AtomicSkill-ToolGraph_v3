FROM python:3.12-slim@sha256:dddfd7e07f9d15aeeca61529320492139d21cac7f0070c00609243e51e4e0016
RUN apt-get update && apt-get install -y --no-install-recommends libreoffice-calc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir openpyxl==3.1.5 pandas==2.3.3 pillow==11.3.0
ENV PYTHONDONTWRITEBYTECODE=1
