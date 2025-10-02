#!/bin/bash

#!/bin/bash

# Install system dependencies first
apt-get update && apt-get install -y tesseract-ocr poppler-utils

# Upgrade pip and install Python dependencies
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Start Streamlit in headless mode
streamlit run main.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
