# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch from its dedicated index. The default PyPI build
# of torch bundles CUDA libraries meant for GPU machines and is several GB;
# the CPU-only build is a fraction of the size and is all we need here.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Download and cache the model weights INSIDE the image at build time.
# This means the container needs no network access at runtime and starts
# up fast; the trade-off is a bigger image and a slower `docker build`.
RUN python -c "from app.model import model; model.load()"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
