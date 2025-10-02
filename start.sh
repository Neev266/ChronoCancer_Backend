#!/bin/bash
set -e

# Start Streamlit in headless mode
exec streamlit run main.py --server.port 8080 --server.address 0.0.0.0 --server.headless true
