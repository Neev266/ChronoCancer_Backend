#!/bin/bash
set -e

# Install system dependencies
apt-get update && apt-get install -y tesseract-ocr poppler-utils

# Upgrade pip and install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start Streamlit
exec streamlit run main.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
