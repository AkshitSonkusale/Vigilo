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

# CORS — allow the deployed frontend (and local dev) to call this API.
# For a portfolio demo we allow all origins; tighten to your Vercel
# URL in production by setting FRONTEND_URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")


@app.get("/")
def root():
    return {"status": "ok", "product": "Vigilo", "version": "0.1.0"}