#!/bin/bash
# Azure App Service (Linux) startup command for Streamlit app
PORT_TO_USE="${PORT:-8000}"

if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing requirements..."
    pip install --no-cache-dir -r requirements.txt
fi

python -m streamlit run app.py \
  --server.port ${PORT_TO_USE} \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false


