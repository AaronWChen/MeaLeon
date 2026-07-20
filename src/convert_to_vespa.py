#!/usr/bin/env python3
"""
src/convert_to_vespa.py

Converts the existing recipe JSON files to Vespa feed format,
adding dense embeddings from the embedding service.

Usage:
    python3 src/convert_to_vespa.py [--limit N] [--source SOURCE]

    --limit N      Only process first N recipes (useful for testing)
    --source       One of: epicurious, allrecipes, bbc, cookstr, all (default: all)

Output files written to data/vespa_feed/

Run from repo root with all services running:
    ML_SERVICE_URL=http://localhost:8000 python3 src/convert_to_vespa.py --limit 100
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from tqdm import tqdm

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8000")
BATCH_SIZE = 32
OUTPUT_DIR = Path("data/vespa_feed")

SOURCE_FILES = {
    "epicurious": "data/recipes-en-201706/epicurious-recipes-to-vespa.json",
    "allrecipes": "data/recipes-en-201706/allrecipes-fixed-urls-uniques-to-vespa.json",
    "bbc": "data/recipes-en-201706/bbc-recipes-to-vespa.json",
    "cookstr": "data/recipes-en-201706/cookstr-recipes-to-vespa.json",
}


def get_embeddings_batch(recipes: list[dict]) -> tuple[list, list]:
    """
    Call embedding service — one request per recipe, since /embed's
    current implementation only correctly processes a single document's
    ingredients per call despite the request schema accepting lists.
    """
    dense_embeddings = []
    bow_embeddings = []

    for recipe in recipes:
        text = f"{recipe['title']}. Ingredients: {', '.join(recipe.get('ingredients', []))}"
        ingredients_str = " ".join(recipe.get("ingredients", []))

        try:
            resp = httpx.post(
                f"{ML_SERVICE_URL}/embed",
                json={
                    "texts": [text],
                    "ingredients": [ingredients_str],
                    "normalize": True,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()

            dense_embeddings.append(data.get("embeddings", [[0.0] * 384])[0])

            bow_raw = data.get("bow_embeddings", {})
            # Flat {term: score} dict for this single document —
            # filter to nonzero, keep top 200 by weight
            bow_filtered = {k: float(v) for k, v in bow_raw.items() if float(v) > 0}
            top = dict(
                sorted(bow_filtered.items(), key=lambda x: x[1], reverse=True)[:200]
            )
            bow_embeddings.append(top)

        except Exception as e:
            print(f"\nWarning: embedding failed for '{recipe.get('title', '?')}': {e}")
            dense_embeddings.append([0.0] * 384)
            bow_embeddings.append({})

    return dense_embeddings, bow_embeddings


def _clean_text(value):
    """
    Strip control characters (except newline/tab) that Vespa's document
    parser rejects. This includes 0x00-0x08, 0x0B-0x1F, 0x7F — the same
    category as the 0x14 that caused the BBC feed failure.
    """
    if isinstance(value, str):
        return re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", value)
    if isinstance(value, list):
        return [_clean_text(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Updated normalise_recipe() — add _clean_text() calls to every string field
# ---------------------------------------------------------------------------


def normalise_recipe(r: dict, source: str) -> dict:
    ingredients = r.get("ingredients", [])
    if isinstance(ingredients, str):
        ingredients = [i.strip() for i in ingredients.split(",") if i.strip()]

    cuisines = r.get("cuisines", r.get("cuisine", []))
    if isinstance(cuisines, str):
        cuisines = [cuisines] if cuisines else []

    return {
        "id": r.get("mealeon_id", r.get("source_id", "")),
        "title": _clean_text(r.get("title", "")),
        "ingredients": _clean_text(ingredients),
        "steps": _clean_text(r.get("steps", [])),
        "cuisine": cuisines,
        "cuisine_str": cuisines[0].lower() if cuisines else "",
        "origin": source,
        "language": r.get("language", "en"),
        "description": _clean_text(r.get("description", "")),
        "url": r.get("url", ""),
        "image_url": r.get("photo_url", ""),
    }


def convert_source(source: str, input_file: str, limit: int | None) -> int:
    """Convert one source file to Vespa feed format. Returns doc count."""
    print(f"\n=== Converting {source} ({input_file}) ===")

    with open(input_file) as f:
        records = json.load(f)

    if limit:
        records = records[:limit]

    print(f"Processing {len(records)} recipes in batches of {BATCH_SIZE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / f"{source}_vespa_feed.json"

    vespa_docs = []
    failed = 0

    for batch_start in tqdm(range(0, len(records), BATCH_SIZE)):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        normalised = [normalise_recipe(r, source) for r in batch]

        # Skip recipes with no ingredients — they can't be meaningfully embedded
        valid = [r for r in normalised if r["ingredients"]]
        if not valid:
            continue

        dense_vecs, bow_vecs = get_embeddings_batch(valid)

        for recipe, dense, bow in zip(valid, dense_vecs, bow_vecs):
            doc_id = recipe["id"] or f"{source}-{batch_start}"
            vespa_docs.append(
                {
                    "put": f"id:recipe:recipe::{source}-{doc_id}",
                    "fields": {
                        **recipe,
                        "embedding": {"values": dense},
                        "ingredients_bow": bow,
                    },
                }
            )

    print(f"Writing {len(vespa_docs)} documents to {output_file}")
    with open(output_file, "w") as f:
        json.dump(vespa_docs, f)

    return len(vespa_docs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit recipes per source (for testing)"
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=list(SOURCE_FILES.keys()) + ["all"],
        help="Which source to convert",
    )
    args = parser.parse_args()

    # Verify embedding service is reachable
    try:
        resp = httpx.get(f"{ML_SERVICE_URL}/health", timeout=5.0)
        resp.raise_for_status()
        print(f"Embedding service: {resp.json()}")
    except Exception as e:
        print(f"ERROR: Embedding service not reachable at {ML_SERVICE_URL}: {e}")
        print(
            "Make sure docker compose is running and embedding_generation is healthy."
        )
        sys.exit(1)

    sources = (
        SOURCE_FILES
        if args.source == "all"
        else {args.source: SOURCE_FILES[args.source]}
    )
    total = 0

    for source, input_file in sources.items():
        if not Path(input_file).exists():
            print(f"Skipping {source} — file not found: {input_file}")
            continue
        count = convert_source(source, input_file, args.limit)
        total += count

    print(f"\nDone. Total documents: {total}")
    print(f"Output files in: {OUTPUT_DIR}/")
    print("\nNext step — feed to Vespa:")
    for source in sources:
        print(f"  vespa feed {OUTPUT_DIR}/{source}_vespa_feed.json")


if __name__ == "__main__":
    main()
