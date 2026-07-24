#!/usr/bin/env python3
"""
src/cuisine_dendrogram_vespa.py

Data exploration: pulls embeddings for all tagged recipes from Vespa,
computes one centroid embedding per cuisine, hierarchically clusters
those centroids, and plots a dendrogram.

This is purely exploratory — it doesn't change any production code.
The goal is to visually inspect which cuisines cluster together based
on your actual trained embeddings, before deciding how to encode that
into an exclusion filter.

Usage:
    pip install scipy matplotlib numpy httpx
    python3 src/cuisine_dendrogram.py
"""

import httpx
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

VESPA_URL = "http://localhost:8080"
BATCH_SIZE = 500

# Placeholders to exclude — same set used in the recommend service's
# cuisine exclusion filter
UNKNOWN_CUISINE_VALUES = {"missing cuisine", "unknown", "n/a", "none", ""}


def fetch_all_tagged_embeddings() -> dict[str, list]:
    """
    Paginate through all Vespa documents with a known cuisine_str,
    collecting embeddings grouped by cuisine.

    Returns: {cuisine: [embedding_vector, ...]}
    """
    cuisine_embeddings = defaultdict(list)
    offset = 0

    while True:
        resp = httpx.post(
            f"{VESPA_URL}/search/",
            json={
                "yql": (
                    "select cuisine_str, embedding from recipe "
                    'where cuisine_str matches ".+" limit '
                    f"{BATCH_SIZE} offset {offset}"
                )
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        hits = resp.json()["root"].get("children", [])

        if not hits:
            break

        for hit in hits:
            fields = hit.get("fields", {})
            cuisine = fields.get("cuisine_str", "")
            embedding = fields.get("embedding", {}).get("values")

            if cuisine in UNKNOWN_CUISINE_VALUES or not embedding:
                continue

            cuisine_embeddings[cuisine].append(embedding)

        offset += BATCH_SIZE
        print(f"Fetched {offset} documents so far...")

        # Safety cap — remove once you've confirmed pagination works,
        # or raise for a full run. Sampling is fine for exploration;
        # you don't need every document to get stable centroids.
        if offset >= 20000:
            print("Reached sample cap (20k docs) — stopping for exploration.")
            break

    return cuisine_embeddings


def compute_centroids(cuisine_embeddings: dict[str, list]) -> tuple[list, np.ndarray]:
    """
    Average embeddings within each cuisine to get one centroid vector.
    Filters out cuisines with too few examples to be meaningful.
    """
    MIN_RECIPES_PER_CUISINE = 20

    labels = []
    centroids = []

    for cuisine, embeddings in sorted(cuisine_embeddings.items()):
        if len(embeddings) < MIN_RECIPES_PER_CUISINE:
            print(
                f"Skipping '{cuisine}' — only {len(embeddings)} recipes (min {MIN_RECIPES_PER_CUISINE})"
            )
            continue
        centroid = np.mean(np.array(embeddings), axis=0)
        labels.append(f"{cuisine} (n={len(embeddings)})")
        centroids.append(centroid)

    return labels, np.array(centroids)


def plot_dendrogram(
    labels: list, centroids: np.ndarray, output_path: str = "cuisine_dendrogram.png"
):
    """
    Hierarchically cluster the centroids and plot a dendrogram.
    Average linkage + cosine-like distance (via correlation) tends to
    work well for high-dimensional embedding centroids.
    """
    Z = linkage(centroids, method="average", metric="cosine")

    plt.figure(figsize=(12, max(8, len(labels) * 0.3)))
    dendrogram(
        Z,
        labels=labels,
        orientation="right",
        leaf_font_size=9,
    )
    plt.title("Cuisine similarity dendrogram (from recipe embeddings)")
    plt.xlabel("Cosine distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved dendrogram to {output_path}")
    print(f"Clustered {len(labels)} cuisines")


def main():
    print("Fetching embeddings grouped by cuisine from Vespa...")
    cuisine_embeddings = fetch_all_tagged_embeddings()

    print(f"\nFound {len(cuisine_embeddings)} distinct cuisine tags")
    for cuisine, embeddings in sorted(
        cuisine_embeddings.items(), key=lambda x: -len(x[1])
    ):
        print(f"  {cuisine}: {len(embeddings)} recipes")

    labels, centroids = compute_centroids(cuisine_embeddings)

    if len(labels) < 3:
        print("Not enough cuisines with sufficient data to cluster meaningfully.")
        return

    plot_dendrogram(labels, centroids)


if __name__ == "__main__":
    main()
