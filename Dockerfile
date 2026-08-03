FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /app/opengym-frontend
COPY opengym-frontend/package*.json ./
RUN npm ci
COPY opengym-frontend/ ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPEN_GYM_DATA_DIR=/var/lib/opengym \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libglew2.2 \
    libgomp1 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /app/opengym-frontend/dist ./opengym-frontend/dist

RUN mkdir -p /var/lib/opengym/models /var/lib/opengym/rollouts

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
