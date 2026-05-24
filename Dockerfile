# Stage 1: Builder
FROM python:3.13-slim AS builder
WORKDIR /app

# 1. Install uv directly from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Copy dependencies configuration
COPY requirements.txt .

# 3. Create a clean virtual environment and install dependencies inside it
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r requirements.txt

# Stage 2: Runtime
FROM python:3.13-slim
WORKDIR /app

# 1. Create a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# 2. Copy the EXACT isolated virtual environment
COPY --from=builder /opt/venv /opt/venv

# 3. Securely append the virtual environment to the system PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# 4. Copy your application source code
COPY --chown=appuser:appgroup . .

USER appuser
CMD ["python", "main.py", "--help"]
