#!/usr/bin/env python3

"""
convert_epicurious_to_vespa.py

Updates the existing convert_epicurious_to_vespa.py to:
  1. Call the embedding_generation service to get dense + sparse embeddings
  2. Add embedding and ingredients_bow fields to each Vespa document
  3. Add cuisine_str (flattened string) for filtering

Usage:
    python convert_epicurious_to_vespa.py <data_dir> <doc_type>

    data_dir: directory containing epicurious-recipes_m2.json
    doc_type: vespa document type, e.g. "recipe"

The ML service URL is read from the ML_SERVICE_URL env var
(default: http://localhost:8000, matching compose.yml).
"""

import json
import os
import sys
import unicodedata
import httpx
from tqdm import tqdm

data_dir = sys.argv[1]
doc_type = sys.argv[2]

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000")
fields = sys.argv[3].split(",")

docs_file = os.path.join(data_dir, "epicurious-recipes_m2.json")
out_file = os.path.join(data_dir, "mealeon_vespa.json")

BATCH_SIZE = (
    32  # how many recipes to embed at once — matches your embedding service batch_size
)

# ── Helpers (kept from original converter) ────────────────────────────────


def fields_populator(k, v):
    lst_keys = ["ingredients", "prepSteps"]

    if k in lst_keys:
        field = (
            "".join(
                cha if unicodedata.category(cha)[0] != "C" else " "
                for sen in v
                for cha in sen + "|"
            )
            .rstrip("|")
            .split("|")
        )

    elif k == "tag":
        field = [
            v["name"] if v["name"] and v["category"] == "cuisine" else "Missing Cuisine"
        ]

    elif k == "id":
        field = f"epicurious-{v}"

    else:
        field = v

    return field


def clean_list_field(value: list) -> list[str]:
    """Remove control characters from list fields (ingredients, steps)."""
    return (
        "".join(
            ch if unicodedata.category(ch)[0] != "C" else " "
            for item in value
            for ch in item + "|"
        )
        .rstrip("|")
        .split("|")
    )


def extract_cuisine(tag: dict) -> list[str]:
    """Extract cuisine from tag field — matches original logic."""
    if tag.get("category") == "cuisine" and tag.get("name"):
        return [tag["name"]]
    return ["Missing Cuisine"]


# ── Embedding service calls ───────────────────────────────────────────────


def get_embeddings_batch(recipes: list[dict]) -> list[dict]:
    """
    Call the embedding_generation service for a batch of recipes.

    Your /embed endpoint takes:
        texts: list[str]       — full text for dense embedding
        ingredients: list[str] — ingredient lists for BoW embedding

    Returns the dense embeddings and BoW sparse maps for each recipe.
    """
    # Build text representations for dense embedding
    # Concatenate title + ingredients — gives the model context about the dish
    texts = [
        f"{r['title']}. Ingredients: {', '.join(r['ingredients'])}" for r in recipes
    ]

    # Ingredients as joined strings for the BoW model
    ingredient_strings = [" ".join(r["ingredients"]) for r in recipes]

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{ML_SERVICE_URL}/embed",
                json={
                    "texts": texts,
                    "ingredients": ingredient_strings,
                    "normalize": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "dense": data["embeddings"],  # list of list[float], shape [batch, 384]
            "sparse": data["bow_embeddings"],  # list of dict[str, float] TF-IDF scores
        }

    except Exception as e:
        print(f"Warning: embedding service error: {e}. Using zero embeddings.")
        # Graceful degradation — return zeros so conversion doesn't fail
        # You can re-run the converter later once the service is up
        return {
            "dense": [[0.0] * 384 for _ in recipes],
            "sparse": [{} for _ in recipes],
        }


def bow_dict_to_vespa_weightedset(bow: dict) -> dict:
    """
    Convert sklearn TF-IDF output dict to Vespa weightedset format.

    Vespa weightedset<string> format: {"term": weight, ...}
    sklearn TF-IDF already produces this shape — just pass it through.
    Filter out zero weights and cap to top-200 terms to keep documents lean.
    """
    if isinstance(bow, dict):
        filtered = {k: float(v) for k, v in bow.items() if float(v) > 0}
        # Keep top 200 by weight
        top = dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:200])
        return top
    return {}


def main():
    # renamer_dict = {
    #     "id": "id",
    #     "hed": "title",
    #     "ingredients": "ingredients",
    #     "prepSteps": "steps",
    #     "tag": "cuisine",
    # }

    print(f"Reading {docs_file}...")
    with open(docs_file) as f:
        records = json.load(f)

    print(f"Processing {len(records)} recipes in batches of {BATCH_SIZE}...")

    output_vespa = []

    for batch_start in tqdm(range(0, len(records), BATCH_SIZE)):
        batch_records = records[batch_start : batch_start + BATCH_SIZE]

        # Build intermediate representation matching original field names
        batch_prepped = []
        for record in batch_records:
            ingredients = clean_list_field(record.get("ingredients", []))
            steps = clean_list_field(record.get("prepSteps", []))
            tag = record.get("tag", {})
            cuisine = extract_cuisine(tag)

            batch_prepped.append(
                {
                    "id": f"epicurious-{record['id']}",
                    "title": record.get("hed", ""),
                    "ingredients": ingredients,
                    "steps": steps,
                    "cuisine": cuisine,
                    "cuisine_str": cuisine[0] if cuisine else "",
                    "origin": "epicurious",
                    "language": "en",
                    "description": "",
                }
            )

        # Get embeddings for this batch
        embeddings = get_embeddings_batch(batch_prepped)

        for i, (prepped, dense_vec, sparse_vec) in enumerate(
            zip(batch_prepped, embeddings["dense"], embeddings["sparse"])
        ):
            vespa_record = {
                "put": f"id:{doc_type}:{doc_type}::{prepped['id']}",
                "fields": {
                    **prepped,
                    # Dense embedding as Vespa tensor values
                    # Vespa expects: {"values": [0.1, 0.2, ...]}
                    "embedding": {"values": dense_vec},
                    # Sparse BoW as weightedset
                    "ingredients_bow": bow_dict_to_vespa_weightedset(sparse_vec),
                },
            }
            output_vespa.append(vespa_record)

    print(f"Writing {len(output_vespa)} documents to {out_file}...")
    with open(out_file, "w") as out:
        json.dump(output_vespa, out)

    print("Done.")


if __name__ == "__main__":
    main()
