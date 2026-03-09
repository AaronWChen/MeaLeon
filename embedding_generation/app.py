from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import mlflow
import mlflow.pytorch
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import os

app = FastAPI()

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load model at startup (keep in memory)
@lru_cache(maxsize=1)
def get_model(model_version: str = "latest"):
    """Load model from MLflow Model Registry or local path"""
    try:
        # Option 1: Load from MLflow Model Registry
        model_name = "embedding-model"

        if model_version == "latest":
            # Get latest production model
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=["Production"])

            if not versions:
                # Fallback to any version
                versions = client.get_latest_versions(model_name)

            if versions:
                model_uri = f"models:/{model_name}/{versions[0].version}"
                print(f"Loading model from MLflow: {model_uri}")
                # Load using MLflow
                model = mlflow.pytorch.load_model(model_uri)
            else:
                raise Exception("No model found in registry")
        else:
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pytorch.load_model(model_uri)

    except Exception as e:
        # Option 2: Fallback to local model (tracked by DVC)
        print(f"MLflow load failed: {e}. Loading from local path.")
        model_path = "/app/models/embeddings/model"

        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
        else:
            # Option 3: Download from Hugging Face
            print("Loading default model from Hugging Face")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model


class EmbeddingRequest(BaseModel):
    texts: List[str]
    normalize: bool = True
    model_version: Optional[str] = "latest"


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int


@app.post("/embed")
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings with specified model version"""
    try:
        model = get_model(request.model_version)

        # Log to MLflow (optional - for monitoring)
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_param("num_texts", len(request.texts))
            mlflow.log_param("model_version", request.model_version)

            embeddings = model.encode(
                request.texts,
                normalize_embeddings=request.normalize,
                batch_size=32,
                show_progress_bar=False,
            )

            mlflow.log_metric("batch_size", len(request.texts))

        return {
            "embeddings": embeddings.tolist(),
            "model_version": request.model_version,
            "dimensions": len(embeddings[0]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/reload-model")
async def reload_model(version: str = "latest"):
    """Reload model from registry (developer endpoint)"""
    get_model.cache_clear()
    model = get_model(version)
    return {
        "status": "reloaded",
        "version": version,
        "device": next(model.parameters()).device.type,
    }


@app.get("/model-info")
async def model_info():
    """Get current model information"""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("embedding-model", stages=["Production"])

        if versions:
            latest = versions[0]
            return {
                "model_name": "embedding-model",
                "version": latest.version,
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "created_at": latest.creation_timestamp,
            }
    except Exception as e:
        return {"error": str(e)}


@app.post("/embed-batch")
async def create_embeddings_batch(texts: List[str]):
    """Batch endpoint for efficiency"""
    model = get_model()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)
    return {"embeddings": embeddings.tolist()}


@app.get("/health")
async def health():
    model = get_model()
    return {
        "status": "healthy",
        "model": model.parameters().name,
        "device": next(model.parameters()).device.type,
        "dimensions": model.get_sentence_embedding_dimension(),
    }


# @app.get("/models")
# async def list_models():
#   """List available models"""
#   return {
#       "current": "sentence-transformers/all-MiniLM-L6-v2",
#       "available": [
#           "all-MiniLM-L6-v2",  *# Fast, 384 dims*
#           "all-mpnet-base-v2",  *# Better quality, 768 dims*
#           "bge-large-en-v1.5"   *# Best quality, 1024 dims*
#       ]
#   }
