import json

WEIGHT_SCALE = 10000

for source in ["epicurious", "allrecipes", "bbc", "cookstr"]:
    path = f"data/vespa_feed/{source}_vespa_feed.json"
    with open(path) as f:
        docs = json.load(f)

    for doc in docs:
        bow = doc["fields"].get("ingredients_bow", {})
        scaled = {term: round(weight * WEIGHT_SCALE) for term, weight in bow.items()}
        doc["fields"]["ingredients_bow"] = {
            term: w for term, w in scaled.items() if w > 0
        }

    with open(path, "w") as f:
        json.dump(docs, f)

    print(f"{source}: rescaled {len(docs)} documents")
