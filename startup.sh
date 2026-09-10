#!/bin/bash
# Azure App Service (Linux) custom startup command for a Streamlit app.
# Set this as the "Startup Command" in App Service > Configuration > General settings:
#   bash startup.sh
python -m streamlit run app.py \
  --server.port 8000 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
