FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install elan (Lean version manager) + Lean
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf \
    | sh -s -- -y && \
    /root/.elan/bin/lean --version

ENV PATH="/root/.elan/bin:${PATH}"

# Python deps (LeanDojo etc.) — uses Python 3.11 inside container
RUN pip install --upgrade pip && \
    pip install "lean-dojo[all]"

# For now, just print Lean version as a health check
CMD ["lean", "--version"]
