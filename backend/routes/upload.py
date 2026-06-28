"""
Upload and sample data route handlers for Vigilo.

POST /api/upload  — accepts a Google Ads CSV file upload, runs the
                    full ML pipeline, returns a VigiloResponse JSON.

GET  /api/sample  — runs the same pipeline on the bundled sample
                    dataset. The frontend calls this on first load so
                    the dashboard is never empty for a new user or
                    recruiter demoing the product.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from models import VigiloResponse
from pipeline import run_pipeline
from utils.csv_parser import ParseError

router = APIRouter()

SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample_campaigns.csv"


@router.post("/upload", response_model=VigiloResponse)
async def upload_csv(file: UploadFile = File(...)) -> VigiloResponse:
    """
    Accepts a Google Ads CSV export, runs the full ML pipeline, and
    returns campaign scores, cluster labels, anomaly flags, health
    scores, and Claude API recommendations.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected a .csv file, got '{file.filename}'. "
                   "Export your campaigns from Google Ads → Reports → "
                   "Predefined reports → Campaign.",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB guard
        raise HTTPException(
            status_code=400,
            detail="File too large. Vigilo supports exports up to 10 MB. "
                   "Try narrowing your date range or filtering to fewer campaigns.",
        )

    try:
        result = run_pipeline(file_bytes)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}. "
                   "Please check your CSV format and try again.",
        )

    return result


@router.get("/sample", response_model=VigiloResponse)
def get_sample_data() -> VigiloResponse:
    """
    Runs the ML pipeline on the bundled sample dataset.
    Called by the frontend on first load to pre-populate the dashboard.
    """
    if not SAMPLE_DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Sample data not found. Please contact support.",
        )

    with open(SAMPLE_DATA_PATH, "rb") as f:
        file_bytes = f.read()

    try:
        result = run_pipeline(file_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run pipeline on sample data: {str(e)}",
        )

    return result
