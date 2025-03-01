# Use a smaller Python image for optimization
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose the correct port for Cloud Run
EXPOSE 8080

# Set environment variable for Google Cloud Run
ENV PORT=8080

# Run the FastAPI application with correct host & port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
