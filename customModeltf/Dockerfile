FROM python:3.10-slim

# Install system dependencies (optional, only if needed later)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Expose port (optional for documentation)
EXPOSE 5000

# Run the server
CMD ["uvicorn", "customModel:app", "--host", "0.0.0.0", "--port", "5000"]
