# Use a smaller Python image for optimization
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# **Install missing dependencies**
RUN pip install python-multipart

# Expose the correct port for Google Cloud Run
EXPOSE 8080

# Set environment variable for Google Cloud Run
ENV PORT=8080

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
