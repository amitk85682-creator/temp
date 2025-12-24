import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app  # Adjust based on your structure
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
