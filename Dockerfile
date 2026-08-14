FROM python:3.10-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.3.1+cpu.html

# Copy the rest
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

# Longer timeout to allow first-request model download from HF Hub
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1"]
