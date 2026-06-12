from contextlib import asynccontextmanager
import asyncio
import os
import re

import dill as pickle
import mlflow
import mlflow.pytorch
import mlflow.pyfunc
import numpy as np
import pandas as pd
import stanza
import torch
from fastapi import FastAPI, HTTPException, Request
from functools import lru_cache
from pydantic import BaseModel, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from typing import List, Optional

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set a short timeout so MLflow connection failures fail fast
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"  # 5 seconds max
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"  # only try once

# ---------------------------------------------------------------------------
# Model loader — restored from commented-out block
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def get_model(model_version: str = "latest", model_name: str = "embedding-model"):
    """
    Load model from MLflow Model Registry with type-aware fallbacks.

    sklearn_transformer fallback chain:
      1. MLflow registry
      2. Local pickle at /app/models/sklearn_transformer.pkl
      3. None — BoW embeddings disabled, routes handle gracefully

    embedding-model fallback chain:
      1. MLflow registry
      2. Local SentenceTransformer at /app/models/embeddings/model
      3. HuggingFace download (all-MiniLM-L6-v2)

    The two model types need completely separate fallback logic —
    a SentenceTransformer cannot substitute for a sklearn TF-IDF model.
    """
    is_sklearn = model_name == "sklearn_transformer"

    # Option 1: MLflow Model Registry
    try:
        client = mlflow.tracking.MlflowClient()
        if model_version == "latest":
            versions = client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                versions = client.get_latest_versions(model_name)
        else:
            versions = [type("v", (), {"version": model_version})]

        if versions:
            model_uri = f"models:/{model_name}/{versions[0].version}"
            print(f"Loading model from MLflow: {model_uri}")
            if is_sklearn:
                return mlflow.pyfunc.load_model(model_uri)
            else:
                return mlflow.pytorch.load_model(model_uri)
        else:
            raise Exception("No model versions found in registry")

    except Exception as e:
        print(f"MLflow load failed ({e}), trying local path...")

    # Option 2 + 3: type-aware fallbacks
    if is_sklearn:
        # sklearn model — try local pickle only, no HuggingFace equivalent
        local_pkl = "/app/models/sklearn_transformer.pkl"
        if os.path.exists(local_pkl):
            print(f"Loading sklearn model from local pickle: {local_pkl}")
            with open(local_pkl, "rb") as f:
                return pickle.load(f)
        else:
            # No local file — return None so service still starts.
            # Routes needing BoW embeddings return empty results rather
            # than crashing the whole service.
            print("WARNING: sklearn_transformer unavailable — BoW embeddings disabled")
            return None
    else:
        # Sentence transformer — local path then HuggingFace
        local_path = "/app/models/embeddings/model"
        if os.path.exists(local_path):
            print(f"Loading embedding model from local path: {local_path}")
            model = SentenceTransformer(local_path)
        else:
            print("Downloading default embedding model from HuggingFace...")
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        return model


# ---------------------------------------------------------------------------
# Lifespan — loads models before the service accepts traffic
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()

    # Load both models concurrently
    print("Loading models concurrently...")
    embedding_future = loop.run_in_executor(
        None, lambda: get_model(model_name="embedding-model")
    )
    sklearn_future = loop.run_in_executor(
        None, lambda: get_model(model_name="sklearn_transformer")
    )
    stanza_future = loop.run_in_executor(
        None,
        lambda: stanza.Pipeline(
            "en",
            processors="tokenize,pos,lemma",
            use_gpu=torch.cuda.is_available(),
            verbose=False,
        ),
    )

    # Wait for all three concurrently
    app.state.embedding_model, app.state.bow_model, app.state.stanza_nlp = (
        await asyncio.gather(embedding_future, sklearn_future, stanza_future)
    )

    if app.state.bow_model is None:
        print(
            "WARNING: BoW model unavailable — /embed will return empty bow_embeddings"
        )

    print("All models loaded — service ready.")
    yield

    print("All models loaded — service ready.")
    yield

    # Cleanup
    del app.state.embedding_model
    del app.state.bow_model
    del app.state.stanza_nlp


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EmbeddingRequest(BaseModel):
    texts: List[str]
    ingredients: List[str]
    normalize: bool = True
    model_version: Optional[str] = "latest"
    bow_model_version: Optional[str] = "latest"


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    # dict instead of pd.DataFrame — FastAPI can serialise this to JSON directly
    bow_embeddings: dict
    bow_model: str
    dimensions: int


# ---------------------------------------------------------------------------
# CustomSKLearnAnalyzer — unchanged from original
# ---------------------------------------------------------------------------


class CustomSKLearnAnalyzer:
    """Handles using Stanza with a custom analyzer inside sklearn."""

    def __init__(self, stanza_lang_str="en"):
        self.stanza_lang_str = stanza_lang_str

    def prepare_stanza_pipeline(
        self,
        depparse_batch_size=50,
        depparse_min_length_to_batch_separately=50,
        verbose=True,
        use_gpu=True,
        batch_size=100,
    ):
        return stanza.Pipeline(
            self.stanza_lang_str,
            depparse_batch_size=depparse_batch_size,
            depparse_min_length_to_batch_separately=depparse_min_length_to_batch_separately,
            verbose=verbose,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

    @classmethod
    def ngram_maker(cls, min_ngram_length: int, max_ngram_length: int):
        def ngrams_per_line(row: str):
            for ln in row.split(" brk "):
                pattern = r"(?u)\b\w{2,}\b"
                terms = re.findall(pattern, ln)
                for ngram_length in range(min_ngram_length, max_ngram_length + 1):
                    for ngram in (
                        word
                        for i in range(len(terms) - ngram_length + 1)
                        for word in (" ".join(terms[i : i + ngram_length]),)
                    ):
                        yield ngram

        return ngrams_per_line


# ---------------------------------------------------------------------------
# CustomSKLearnPythonModel — unchanged from original
# ---------------------------------------------------------------------------


class CustomSKLearnPythonModel(mlflow.pyfunc.PythonModel):
    """Allows Stanza pipelines to be logged in MLflow as a custom PythonModel."""

    def load_context(self, context):
        with open(context.artifacts["sklearn_model"], "rb") as f:
            self.model = pickle.load(f)
        with open(context.artifacts["sklearn_transformer"], "rb") as f:
            self.sklearn_transformer = pickle.load(f)

    def predict(self, context, model_input):
        response = self.sklearn_transformer.transform(model_input.values)
        return pd.DataFrame(
            response.toarray(),
            columns=self.sklearn_transformer.get_feature_names_out(),
            index=model_input.index,
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="MeaLeon Embedding Service", lifespan=lifespan)


@app.get("/health")
async def health(request: Request):
    """
    Health check — confirms models are loaded and service is ready.
    Fixed from original: was calling undefined get_model() and using
    model.parameters().name which doesn't exist on SentenceTransformer.
    """
    embedding_model: SentenceTransformer = request.app.state.embedding_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "healthy",
        "device": device,
        # SentenceTransformer doesn't have .parameters().name —
        # use get_sentence_embedding_dimension() instead
        "dimensions": embedding_model.get_sentence_embedding_dimension(),
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: Request, body: EmbeddingRequest):
    """
    Generate dense + sparse embeddings for a batch of texts.

    Fixed from original:
      - was reading request.texts (raw Request) instead of body.texts (Pydantic model)
      - bow_model key was inconsistent between lifespan and route
      - DataFrame returned directly (not JSON-serialisable) — now converted to dict
    """
    embedding_model: SentenceTransformer = request.app.state.embedding_model
    bow_model = request.app.state.bow_model

    try:
        embeddings = embedding_model.encode(
            body.texts,
            normalize_embeddings=body.normalize,
            batch_size=32,
            show_progress_bar=False,
        )

        # BoW embeddings — gracefully disabled if sklearn model unavailable
        if bow_model is not None:
            bow_df: pd.DataFrame = bow_model.predict(
                pd.Series(body.ingredients, name="ingredients")
            )
            bow_dict = bow_df.to_dict(orient="list")
        else:
            bow_dict = {}

        # if MLflow is available
        try:
            with mlflow.start_run(run_name="inference", nested=True):
                mlflow.log_param("num_texts", len(body.texts))
                mlflow.log_param("model_version", body.model_version)
                mlflow.log_param("bow_model_version", body.bow_model_version)

                mlflow.log_metric("batch_size", len(body.texts))
        except Exception as mlflow_err:
            print(f"MLflow tracking skipped: {mlflow_err}")

        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model=body.model_version or "latest",
            bow_embeddings=bow_dict,
            bow_model=body.bow_model_version or "latest",
            dimensions=len(embeddings[0]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/embed-batch")
async def embed_batch(request: Request, texts: List[str]):
    """
    Lightweight batch endpoint — dense embeddings only, no BoW.
    Fixed from original: was calling undefined get_model() directly.
    """
    embedding_model: SentenceTransformer = request.app.state.embedding_model
    embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=False)
    return {"embeddings": embeddings.tolist()}


@app.get("/model-info")
async def model_info():
    """Get current model information from MLflow registry."""
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
        return {"model_name": "embedding-model", "version": "none in Production stage"}
    except Exception as e:
        return {"error": str(e)}
