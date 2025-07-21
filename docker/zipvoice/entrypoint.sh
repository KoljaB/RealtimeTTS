#!/bin/bash
echo "Starting FastAPI ZipVoice Server..."

# Navigate to the application directory.
cd /opt/app-root/src/

# Execute the FastAPI server script using exec.
# This will start the Uvicorn server defined inside the Python script.
exec python3 ./serve_zipchunks.py