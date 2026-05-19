from fastapi import FastAPI, HTTPException
import dill as pickle
from pydantic import BaseModel, ConfigDict, field_validator
from typing import List, Optional
import mlflow
import mlflow.pytorch
import torch
from sentence_transformers import SentenceTransformer
import stanza

# from
import numpy as np
import pandas as pd
from functools import lru_cache
import os

app = FastAPI()

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load model at startup (keep in memory)
@lru_cache(maxsize=1)
def get_model(model_version: str = "latest", model_name: str = "embedding-model"):
    """Load model from MLflow Model Registry or local path"""
    try:
        # Option 1: Load from MLflow Model Registry
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
                if model_name == "sklearn_transformer":
                    model = mlflow.pyfunc.load_model(model_uri)
                else:
                    model = mlflow.pytorch.load_model(model_uri)
            else:
                raise Exception("No model found in registry")
        else:
            model_uri = f"models:/{model_name}/{model_version}"
            if model_name == "sklearn_transformer":
                model = mlflow.pyfunc.load_model(model_uri)
            else:
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
    ingredients: List[str]
    normalize: bool = True
    model_version: Optional[str] = "latest"
    bow_model_version: Optional[str] = "latest"


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: List[List[float]]
    model: str
    bow_embeddings: pd.DataFrame
    bow_model: str
    dimensions: int

    @field_validator("bow_embeddings")
    def check_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise TypeError("bow_embeddings must be a pandas DataFrame")

        return v


class CustomSKLearnAnalyzer:
    """
    This class handles using Stanza with a custom analyzer inside sklearn
    """

    def __init__(self, stanza_lang_str="en"):
        """
        Constructor method. Initializes the model with a Stanza libary language
        type. The default is "en" for English, later on, can think adding
        functionality to download the pretrained model/embeddings
        """
        self.stanza_lang_str = stanza_lang_str

    def prepare_stanza_pipeline(
        self,
        depparse_batch_size=50,
        depparse_min_length_to_batch_separately=50,
        verbose=True,
        use_gpu=True,
        batch_size=100,
    ):
        """
        Method to simply construction of Stanza Pipeline for usage in the sklearn custom analyzer

        Args:
            Follow creation of stanza pipeline (link to their docs)

            self.stanza_lang_str:
                str for pretrained Stanza embeddings to use in the pipeline (from init)

            depparse_batch_size:
                int for batch size for processing, default is 50

            depparse_min_length_to_batch_separately:
                int for minimum string length to batch, default is 50

            verbose:
                boolean for information for readouts during processing, default is True

            use_gpu:
                boolean for using GPU for stanza, default is False,
                set to True when on cloud/not on streaming computer

            batch_size:
                int for batch sizing, default is 100

        Returns:
            nlp:
                stanza pipeline
        """

        nlp = stanza.Pipeline(
            self.stanza_lang_str,
            depparse_batch_size=depparse_batch_size,
            depparse_min_length_to_batch_separately=depparse_min_length_to_batch_separately,
            verbose=verbose,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

        return nlp

    @classmethod
    def ngram_maker(self, min_ngram_length: int, max_ngram_length: int):
        def ngrams_per_line(row: str):
            for ln in row.split(" brk "):
                at_least_two_english_characters_whole_words = r"(?u)\b\w{2,}\b"
                terms = re.findall(at_least_two_english_characters_whole_words, ln)
                for ngram_length in range(min_ngram_length, max_ngram_length + 1):

                    # find and return all ngrams
                    # for ngram in zip(*[terms[i:] for i in range(3)]):
                    # <-- solution without a generator (works the same but has higher memory usage)
                    for ngram in (
                        word
                        for i in range(len(terms) - ngram_length + 1)
                        for word in (" ".join(terms[i : i + ngram_length]),)
                    ):
                        yield ngram

        return ngrams_per_line


class CustomSKLearnPythonModel(mlflow.pyfunc.PythonModel):
    """
    This class allows Stanza pipelines to be logged in MLflow as a
    custom PythonModel
    """

    # def __init__(self, model):
    #     """
    #     Constructor method. Initializes the model with a Stanza libary language
    #     type. The default is "en" for English

    #     model:          sklearn.Transformer
    #             The sklearn text Transformer or Pipeline that ends in a
    #             Transformer

    #     later can add functionality to include pretrained models needed for Stanza

    #     """
    #     self.model = model

    def load_context(self, context):
        """
        Method needed to override default load_context. Needs to handle different components of sklearn model
        """
        with open(context.artifacts["sklearn_model"], "rb") as f:
            self.model = pickle.load(f)

        with open(context.artifacts["sklearn_transformer"], "rb") as f:
            self.sklearn_transformer = pickle.load(f)

    def predict(self, model_input):
        """
        This method is needed to override the default predict.
        It needs to function essentially as a wrapper and returns back the
        Transformer or Transformer Pipeline itself

        Args:
            context:        Any
                Not used

            model:          sklearn.Transformer
                The sklearn text Transformer or Pipeline that ends in a
                Transformer

            model_input:    List(string)
                The ingredients of a single query recipe in a list
                Need to decide if this is taking in raw text or preprocessed text
                Leaning towards taking in raw text, doing preprocessing, and
                logging the pre processed text as an artifact

            params:         dict, optional
                Parameters used for the model (optional)
                Not used currently for sklearn

        Returns:
            transformed_recipe_df: DataFrame of the recipes after going through
            the sklearn/Stanza text processing
        """
        response = self.sklearn_transformer.transform(model_input.values)

        transformed_recipe = pd.DataFrame(
            response.toarray(),
            columns=self.sklearn_transformer.get_feature_names_out(),
            index=model_input.index,
        )

        return transformed_recipe

    def encode(self, context, texts_to_encode, params):
        # cv_params are parameters for the sklearn CountVectorizer or TFIDFVectorizer
        sklearn_transformer_params = {
            "strip_accents": "unicode",
            "lowercase": True,
            "analyzer": CustomSKLearnAnalyzer().stanza_analyzer(
                stanza_pipeline=nlp, minNgramLength=1, maxNgramLength=4
            ),
            "min_df": 3,
            # 'binary':False
        }

        # pipeline_params are parameters that will be logged in MLFlow and are a superset of library parameters
        pipeline_params = {"stanza_model": "en", "sklearn-transformer": "TFIDF"}

        # update the pipeline parameters with the library-specific ones so that they show up in MLflow Tracking
        pipeline_params.update(sklearn_transformer_params)

        # Instantiate sklearn transformer
        sklearn_transformer = TfidfVectorizer(**sklearn_transformer_params)

        # Do fit transform on data
        response = sklearn_transformer.fit_transform(tqdm(model_input))

        signature = infer_signature(
            model_input=model_input, model_output=transformed_recipe
        )

        with open(sklearn_transformer_path, "wb") as fo:
            pickle.dump(sklearn_transformer, fo)

        with open(transformed_recipes_path, "wb") as fo:
            pickle.dump(transformed_recipe, fo)

        model_info = mlflow.pyfunc.log_model(
            code_path=["../src/"],
            python_model=CustomSKLearnWrapper(),
            input_example=to_nlp_df["ingredients"][0],
            signature=signature,
            artifact_path="sklearn_model",
            artifacts=artifacts,
        )


@app.post("/embed")
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings with specified model version"""
    try:
        embedding_model = get_model(request.model_version, model_name="embedding-model")

        bow_embedding_model: CustomSKLearnPythonModel = get_model(
            request.bow_model_version, model_name="sklearn_transformer"
        )

        # Log to MLflow (optional - for monitoring)
        with mlflow.start_run(run_name="inference", nested=True):
            mlflow.log_param("num_texts", len(request.texts))
            mlflow.log_param("model_version", request.model_version)
            mlflow.log_param("bow_model_version", request.bow_model_version)

            embeddings = embedding_model.encode(
                request.texts,
                normalize_embeddings=request.normalize,
                batch_size=32,
                show_progress_bar=False,
            )

            bow_embeddings = bow_embedding_model.predict(request.ingredients)

            mlflow.log_metric("batch_size", len(request.texts))

        return {
            "embeddings": embeddings.tolist(),
            "model_version": request.model_version,
            "bow_embeddings": bow_embeddings,
            "bow_model_version": request.bow_model_version,
            "dimensions": len(embeddings[0]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


# @app.post("/bow-vectorizer")
# async def create_bow_embeddings(request: EmbeddingRequest) -> dict[str, Any]:
#     """Generate bag-of-words embeddings with specified version"""
#     try:
#         model:


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
