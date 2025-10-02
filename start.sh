#!/bin/bash

# Install dependencies
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Start Streamlit in headless mode for Railway
streamlit run main.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
