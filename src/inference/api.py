"""
REST API for music structure recognition.

Provides HTTP endpoints for predicting structure from uploaded audio files.
"""

import os
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import numpy as np
import librosa
import soundfile as sf

from modeling.system import MusicStructureModel, ModelConfig
from data.preprocessing import AudioPreprocessor
from inference.cli import InferenceConfig, MusicStructureInference

logger = logging.getLogger(__name__)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for prediction."""

    track_id: Optional[str] = None
    threshold: float = 0.5
    model_type: str = "mert"


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    track_id: str
    sr: int
    duration: float
    boundaries: List[float]
    labels: List[str]
    segments: List[Dict[str, Any]]
    version: str
    beats: Optional[List[float]] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None


# Global inference engine
inference_engine: Optional[MusicStructureInference] = None


def create_app(model_path: str) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Music Structure Recognition API",
        description="API for detecting musical structure boundaries and labels",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize inference engine
    @app.on_event("startup")
    async def startup_event():
        global inference_engine
        try:
            config = InferenceConfig(model_path=model_path)
            inference_engine = MusicStructureInference(config)
            logger.info("Inference engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize inference engine: {e}")
            raise

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Music Structure Recognition API",
            "version": "1.0.0",
            "endpoints": {"predict": "/predict", "health": "/health", "docs": "/docs"},
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if inference_engine is None:
            raise HTTPException(
                status_code=503, detail="Inference engine not initialized"
            )

        return {
            "status": "healthy",
            "model_loaded": True,
            "device": str(inference_engine.device),
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict_structure(
        background_tasks: BackgroundTasks,
        audio_file: UploadFile = File(...),
        threshold: float = 0.5,
        track_id: Optional[str] = None,
    ):
        """
        Predict music structure from uploaded audio file.

        Args:
            audio_file: Uploaded audio file
            threshold: Boundary detection threshold
            track_id: Optional track identifier

        Returns:
            Prediction results
        """
        if inference_engine is None:
            raise HTTPException(
                status_code=503, detail="Inference engine not initialized"
            )

        # Validate file type
        if not audio_file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload WAV, MP3, FLAC, or M4A files.",
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(audio_file.filename).suffix
        ) as tmp_file:
            try:
                # Save uploaded file
                content = await audio_file.read()
                tmp_file.write(content)
                tmp_file.flush()

                # Predict structure
                result = inference_engine.predict(tmp_file.name)

                # Update track_id if provided
                if track_id:
                    result["track_id"] = track_id

                # Clean up temporary file
                background_tasks.add_task(os.unlink, tmp_file.name)

                return PredictionResponse(**result)

            except Exception as e:
                # Clean up on error
                background_tasks.add_task(os.unlink, tmp_file.name)
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Prediction failed: {str(e)}"
                )

    @app.post("/predict-batch")
    async def predict_batch(
        background_tasks: BackgroundTasks,
        audio_files: List[UploadFile] = File(...),
        threshold: float = 0.5,
    ):
        """
        Predict music structure for multiple audio files.

        Args:
            audio_files: List of uploaded audio files
            threshold: Boundary detection threshold

        Returns:
            List of prediction results
        """
        if inference_engine is None:
            raise HTTPException(
                status_code=503, detail="Inference engine not initialized"
            )

        if len(audio_files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400, detail="Too many files. Maximum 10 files per batch."
            )

        results = []
        temp_files = []

        try:
            # Save all files temporarily
            for audio_file in audio_files:
                if not audio_file.filename.lower().endswith(
                    (".wav", ".mp3", ".flac", ".m4a")
                ):
                    continue  # Skip unsupported files

                tmp_file = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(audio_file.filename).suffix
                )
                content = await audio_file.read()
                tmp_file.write(content)
                tmp_file.flush()
                temp_files.append(tmp_file.name)

            # Process all files
            for i, temp_file in enumerate(temp_files):
                try:
                    result = inference_engine.predict(temp_file)
                    result["track_id"] = audio_files[i].filename
                    results.append(result)
                except Exception as e:
                    results.append(
                        {"track_id": audio_files[i].filename, "error": str(e)}
                    )

            # Clean up temporary files
            for temp_file in temp_files:
                background_tasks.add_task(os.unlink, temp_file)

            return {"results": results}

        except Exception as e:
            # Clean up on error
            for temp_file in temp_files:
                background_tasks.add_task(os.unlink, temp_file)
            logger.error(f"Batch prediction failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Batch prediction failed: {str(e)}"
            )

    @app.get("/models")
    async def list_models():
        """List available models."""
        return {
            "available_models": ["mert", "w2v"],
            "current_model": inference_engine.config.model_type
            if inference_engine
            else None,
        }

    @app.get("/config")
    async def get_config():
        """Get current configuration."""
        if inference_engine is None:
            raise HTTPException(
                status_code=503, detail="Inference engine not initialized"
            )

        return {
            "model_path": inference_engine.config.model_path,
            "model_type": inference_engine.config.model_type,
            "device": str(inference_engine.device),
            "threshold": inference_engine.config.threshold,
        }

    return app


def main():
    """Main function to run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Music Structure Recognition REST API")
    parser.add_argument(
        "--model", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create app
    app = create_app(args.model)

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
