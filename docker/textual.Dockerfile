FROM python:3.11-slim

# System dependencies for clingo etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command: run ASP smoke test to check everything works
CMD ["python", "-m", "asp.smoke_test"]
