FROM python:3.12-slim

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY i18n ./i18n

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

# Cloud Run provides $PORT; single worker is enough at v1.0 traffic.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
