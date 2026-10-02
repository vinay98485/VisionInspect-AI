FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install the rest
COPY backend/requirements.txt ./backend/
RUN --mount=type=cache,target=/root/.cache/pip pip install --extra-index-url https://download.pytorch.org/whl/cpu -r backend/requirements.txt

# Copy tracked source files
COPY backend/ ./backend/
COPY ai/ ./ai/

# Set working directory for python execution
WORKDIR /app/backend

EXPOSE 8000

CMD ["sh", "-c", "python -m app.database.init_db && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
