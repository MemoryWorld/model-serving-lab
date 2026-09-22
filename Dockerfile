FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY serving_lab ./serving_lab
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home app
USER 10001
EXPOSE 8788
HEALTHCHECK --interval=10s --timeout=2s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8788/healthz',timeout=1)"
# One worker: admission and quota state are intentionally per process.
CMD ["uvicorn", "serving_lab.app:app", "--host", "0.0.0.0", "--port", "8788", "--workers", "1", "--no-access-log", "--timeout-graceful-shutdown", "10"]
