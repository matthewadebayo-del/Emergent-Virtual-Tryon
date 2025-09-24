import uvicorn
import os
from production_server import app

# Ensure environment variables are set
if not os.getenv("MONGO_URL"):
    os.environ["MONGO_URL"] = "mongodb://localhost:27017"
if not os.getenv("DB_NAME"):
    os.environ["DB_NAME"] = "test_database"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)