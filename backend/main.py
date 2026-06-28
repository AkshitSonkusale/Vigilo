"""
Vigilo backend — FastAPI application entry point.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router

app = FastAPI(
    title="Vigilo API",
    description="ML-powered Google Ads campaign optimization",
    version="0.1.0",
)

# Allow local dev + deployed Vercel frontend
# Add your Vercel URL here once deployed e.g. https://vigilo.vercel.app
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5500",
    os.environ.get("FRONTEND_URL", ""),   # set on Render dashboard
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in ALLOWED_ORIGINS if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "product": "Vigilo", "version": "0.1.0"}