# Use a lightweight Python base
FROM python:3.10-slim


# Install Git (and any other OS packages)
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default command: launch the CLI
CMD ["python", "src/cli.py"]
