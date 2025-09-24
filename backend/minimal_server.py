from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="VirtualFit API - Minimal")

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
        "message": "Minimal server for testing - full 3D pipeline disabled",
        "components": {
            "server": "✅ Running",
            "cors": "✅ Enabled",
            "3d_pipeline": "❌ Disabled (use full server for 3D features)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)