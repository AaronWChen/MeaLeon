#!/usr/bin/env python3
"""
src/build_cuisine_clusters.py

Uses Vespa's Document API v1 /visit endpoint (not /search) to iterate
through ALL documents for each cuisine, with no 1000-result ceiling —
visit is designed for exhaustive iteration via continuation tokens,
unlike /search which is bounded for query-serving performance reasons.

Usage:
    pip install scipy numpy httpx
    python3 src/build_cuisine_clusters.py --threshold 0.065
"""

import argparse
import json
import httpx
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster

VESPA_URL = "http://localhost:8080"
VISIT_PAGE_SIZE = 200  # documents per visit page — Vespa recommends starting modest

UNKNOWN_CUISINE_VALUES = {"missing cuisine", "unknown", "n/a", "none", ""}
MIN_RECIPES_PER_CUISINE = 20


def discover_cuisines() -> list[str]:
    """Get the full list of distinct cuisine_str values via Vespa grouping."""
    resp = httpx.post(
        f"{VESPA_URL}/search/",
        json={
            "yql": (
                "select cuisine_str from recipe "
                'where cuisine_str matches ".+" '
                "| all(group(cuisine_str) max(200) each(output(count())))"
            )
        },
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    cuisines = []
    groups = data["root"]["children"][0]["children"][0]["children"]
    for g in groups:
        cuisines.append(g["value"])
    return cuisines


def fetch_all_embeddings_for_cuisine(cuisine: str) -> list:
    """
    Exhaustively fetch every embedding for a given cuisine using the
    Document API visit endpoint with continuation tokens.

    Selection syntax: recipe.cuisine_str=="<value>" — Vespa's document
    selection language, distinct from YQL used by /search.
    """
    embeddings = []
    continuation = None
    # Escape any embedded double quotes defensively
    safe_cuisine = cuisine.replace('"', '\\"')

    while True:
        params = {
            "selection": f'recipe.cuisine_str=="{safe_cuisine}"',
            "wantedDocumentCount": VISIT_PAGE_SIZE,
            "fieldSet": "recipe:embedding",  # only fetch the field we need
        }
        if continuation:
            params["continuation"] = continuation

        resp = httpx.get(
            f"{VESPA_URL}/document/v1/recipe/recipe/docid/",
            params=params,
            timeout=60.0,
        )
        if resp.status_code != 200:
            print(f"    visit error: {resp.text}")
            break

        data = resp.json()
        documents = data.get("documents", [])

        for doc in documents:
            embedding = doc.get("fields", {}).get("embedding", {}).get("values")
            if embedding:
                embeddings.append(embedding)

        continuation = data.get("continuation")
        if not continuation:
            break

    return embeddings


def compute_centroids() -> tuple[list, np.ndarray, dict]:
    print("Discovering distinct cuisines...")
    cuisines = discover_cuisines()
    print(f"Found {len(cuisines)} distinct cuisines\n")

    labels = []
    centroids = []
    counts = {}

    for cuisine in sorted(cuisines):
        if cuisine in UNKNOWN_CUISINE_VALUES:
            continue

        print(f"Visiting all documents for '{cuisine}'...")
        embeddings = fetch_all_embeddings_for_cuisine(cuisine)
        counts[cuisine] = len(embeddings)
        print(f"  {cuisine}: {len(embeddings)} embeddings (full corpus)")

        if len(embeddings) < MIN_RECIPES_PER_CUISINE:
            print(f"  skipping — below minimum of {MIN_RECIPES_PER_CUISINE}")
            continue

        centroid = np.mean(np.array(embeddings), axis=0)
        labels.append(cuisine)
        centroids.append(centroid)

    return labels, np.array(centroids), counts


def build_clusters(labels: list, centroids: np.ndarray, threshold: float) -> dict:
    Z = linkage(centroids, method="average", metric="cosine")
    cluster_ids = fcluster(Z, t=threshold, criterion="distance")
    return {cuisine: int(cid) for cuisine, cid in zip(labels, cluster_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.065)
    parser.add_argument("--output", default="cuisine_clusters.json")
    args = parser.parse_args()

    labels, centroids, counts = compute_centroids()

    if len(labels) < 2:
        print("Not enough cuisines with sufficient data to cluster.")
        return

    cluster_map = build_clusters(labels, centroids, args.threshold)

    by_cluster = defaultdict(list)
    for cuisine, cid in cluster_map.items():
        by_cluster[cid].append(cuisine)

    print(f"\n=== Clusters at threshold {args.threshold} ===")
    for cid, members in sorted(by_cluster.items()):
        print(f"Cluster {cid}: {', '.join(sorted(members))}")

    output = {
        "threshold": args.threshold,
        "cuisine_to_cluster": cluster_map,
        "cluster_members": {str(k): sorted(v) for k, v in by_cluster.items()},
        "recipe_counts": counts,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
