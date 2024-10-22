#!/usr/bin/env python3

import json
import os
import sys
import unicodedata

data_dir = sys.argv[1]
doc_type = sys.argv[2]
fields = sys.argv[3].split(",")

docs_file = os.path.join(data_dir, "epicurious-recipes_m2.json")
out_file = os.path.join(data_dir, "mealeon_vespa.json")


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


def main():
    renamer_dict = {
        "id": "id",
        "hed": "title",
        "ingredients": "ingredients",
        "prepSteps": "steps",
        "tag": "cuisine",
    }

    with open(docs_file) as f, open(out_file, "w") as out:
        d = json.load(f)

        output_vespa = []
        for record in d:
            vespa_record = {
                "put": f"id:{doc_type}:{doc_type}::epicurious-{record['id']}",
                "fields": {
                    "origin": "epicurious",
                    "id": "",
                    "title": "",
                    "ingredients": "",
                    "steps": "",
                    "cuisine": "",
                },
            }

            vespa_record["fields"].update(
                {
                    renamer_dict[k]: fields_populator(k, v)
                    for k, v in record.items()
                    if k in renamer_dict.keys()
                }
            )

            output_vespa.append(vespa_record)

        json.dump(output_vespa, out)


if __name__ == "__main__":
    main()
