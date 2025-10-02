#!/bin/bash

# Install dependencies
pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Start the app
uvicorn chrono_cancer_backend.main:app --host 0.0.0.0 --port 8080
