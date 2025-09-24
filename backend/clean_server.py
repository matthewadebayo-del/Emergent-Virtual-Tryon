from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="VirtualFit API - Clean")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "VirtualFit API is running"}

@app.get("/debug")
async def debug():
    return {
        "status": "Server is running",
        "message": "Clean server without Unicode issues",
        "environment": {
            "MONGO_URL": os.getenv("MONGO_URL", "not_set"),
            "DB_NAME": os.getenv("DB_NAME", "not_set"),
            "FAL_KEY": "configured" if os.getenv("FAL_KEY") else "not_set",
            "OPENAI_API_KEY": "configured" if os.getenv("OPENAI_API_KEY") else "not_set"
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)