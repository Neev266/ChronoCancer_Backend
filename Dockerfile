# Use an official Python base image
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Set working directory
WORKDIR /app

# Copy Python dependency files
COPY requirements.txt .

# Install system dependencies (Tesseract + Poppler for OCR)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

# Make start.sh executable
RUN chmod +x start.sh

# Expose the port Streamlit will run on
EXPOSE 8080

# Start the app
CMD ["bash", "start.sh"]
