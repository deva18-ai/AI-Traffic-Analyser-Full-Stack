"""
AI Traffic Analysis System - FastAPI Backend
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from database.db import init_db
from routes.upload import router as upload_router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Database initialized.")
    yield

app = FastAPI(
    title="AI Traffic Analysis System",
    description="Detect vehicles, classify types, and identify traffic violations using YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)




@app.get("/health")
def health():
    return {"status": "ok", "service": "AI Traffic Analysis System"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
