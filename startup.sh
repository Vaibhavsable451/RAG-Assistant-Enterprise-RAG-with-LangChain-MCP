#!/bin/bash
# Azure App Service (Linux) startup command for Streamlit.
# File uploads fail with Axios HTTP 400 behind Azure's reverse proxy unless
# XSRF/CORS are disabled here. Do not rely only on .streamlit/config.toml:
# GitHub Actions upload-artifact v4 omits hidden folders by default.
PORT_TO_USE="${PORT:-8000}"

export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing requirements..."
    pip install --no-cache-dir -r requirements.txt
fi

python -m streamlit run app.py \
  --server.port "${PORT_TO_USE}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableXsrfProtection false \
  --server.enableCORS false \
  --server.enableWebsocketCompression false \
  --server.maxUploadSize 500
