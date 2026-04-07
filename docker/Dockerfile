# =============================================================================
# CovalentAgent Dockerfile
# Multi-stage build: builder -> runtime
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build-time system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency specification first (layer caching)
COPY pyproject.toml ./

# Create a minimal src layout so pip can resolve the package
COPY src/ ./src/

# Install the package and all dependencies into a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e "."

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim

LABEL maintainer="Harshit Ghosh <harshitghosh@gmail.com>"
LABEL description="CovalentAgent: multi-agent system for covalent drug design"

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system dependencies:
#   libxrender1, libxext6  - required by rdkit-pypi for 2D depiction
#   curl                   - used by the HEALTHCHECK
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user
RUN groupadd --gid 1000 covalent && \
    useradd --uid 1000 --gid covalent --create-home --shell /bin/bash covalent

WORKDIR /app

# Copy application source
COPY --from=builder /build/src ./src
COPY --from=builder /build/pyproject.toml ./

# Copy the data directory (warheads.json, reactive_residues.json)
COPY data/ ./data/

# Copy the Streamlit demo
COPY app/ ./app/

# Re-install in editable mode so the package is importable
RUN pip install --no-cache-dir -e "."

# Create directories for runtime data
RUN mkdir -p /app/chroma_data /app/models/cached && \
    chown -R covalent:covalent /app

# Switch to non-root user
USER covalent

# Environment variables (override at runtime via .env or docker-compose)
ENV ANTHROPIC_API_KEY="" \
    HF_TOKEN="" \
    PUBMED_API_KEY="" \
    CHROMA_PERSIST_DIR="/app/chroma_data" \
    ESM_MODEL="facebook/esm2_t33_650M_UR50D" \
    CHEMPROP_MODEL_DIR="/app/models/cached" \
    PYTHONUNBUFFERED=1

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck: Streamlit serves a health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run the Streamlit demo
CMD ["streamlit", "run", "app/demo.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
