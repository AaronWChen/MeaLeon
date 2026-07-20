#!/usr/bin/env python3
"""
Logs your trained CustomSKLearnPythonModel wrapper to MLflow.

Run on your host (not in a container) with MLflow reachable:
    pip install mlflow scikit-learn stanza dill pandas
    MLFLOW_TRACKING_URI=http://localhost:5001 python3 scripts/register_bow_model.py \
        --transformer-pkl your_trained_transformer.pkl

This does NOT need your training data or the Stanza pipeline present at
log time — only the fitted sklearn_transformer pickle. Stanza gets
re-initialized fresh at MODEL LOAD time (inside the embedding_generation
container), per custom_model.py's load_context().
"""

import argparse
import os
import sys

import mlflow
import mlflow.pyfunc

# Import the model class — must be importable, and its source file
# is passed via code_path so MLflow can reconstruct it in other envs.
sys.path.insert(0, os.path.dirname(__file__))
from custom_model import CustomSKLearnPythonModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transformer-pkl",
        required=True,
        help="Path to your trained sklearn_transformer pickle (dill format)",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"),
    )
    parser.add_argument(
        "--stage",
        default="Production",
        help="Model registry stage to promote to after logging",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    print(f"Logging to MLflow at {args.tracking_uri}")

    artifacts = {
        "sklearn_transformer": args.transformer_pkl,
    }

    with mlflow.start_run(run_name="tfidf-bow-registration") as run:
        mlflow.pyfunc.log_model(
            artifact_path="sklearn_transformer",
            python_model=CustomSKLearnPythonModel(),
            artifacts=artifacts,
            registered_model_name="sklearn_transformer",
            # code_path ensures the class definition travels with the
            # model artifact — required since it's a custom PythonModel,
            # not a built-in flavor.
            code_paths=[os.path.join(os.path.dirname(__file__), "custom_model.py")],
            pip_requirements=[
                "scikit-learn",
                "stanza",
                "dill",
                "pandas",
                "mlflow",
            ],
        )
        run_id = run.info.run_id
        print(f"Logged model in run: {run_id}")

    # Promote to the requested stage so get_model()'s
    # get_latest_versions(..., stages=["Production"]) finds it
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions("sklearn_transformer")
    if not versions:
        print("ERROR: no versions found after logging — something went wrong")
        sys.exit(1)

    latest = versions[0]
    client.transition_model_version_stage(
        name="sklearn_transformer",
        version=latest.version,
        stage=args.stage,
    )
    print(f"Promoted version {latest.version} to stage '{args.stage}'")
    print("\nDone. Restart embedding_generation to pick up the new model:")
    print("  docker compose restart embedding_generation")


if __name__ == "__main__":
    main()
