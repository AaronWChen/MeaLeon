#!/usr/bin/env python3

import json
import os
import sys
import unicodedata

data_dir = sys.argv[1]
doc_type = sys.argv[2]
# fields = sys.argv[3].split(",")

docs_file = os.path.join(data_dir, "postgres_table_dump.json")
out_file = os.path.join(data_dir, "postgres_mealeon_vespa.json")


def unicode_cleaner(k, v):
    lst_keys = ["ingredients", "steps", "cuisines"]
    str_keys = ["title", "description"]

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

    elif k in str_keys:
        field = "".join(
            cha if unicodedata.category(cha)[0] != "C" else " "
            for sen in v
            for cha in sen
        )

    else:
        field = v

    return field


def main():
    with open(docs_file) as f, open(out_file, "w") as out:
        d = json.load(f)

        output_vespa = []
        for record in d:
            vespa_record = {
                "put": f"id:{doc_type}:{doc_type}::{record['mealeon_id']}",
                "fields": {
                    "id": record["mealeon_id"],
                    "title": record["title"],
                    "ingredients": record["ingredients"],
                    "steps": record["steps"],
                    "cuisines": record["cuisines"],
                    "language": record["language"],
                    "description": record["description"],
                },
            }

            vespa_record["fields"].update(
                {
                    k: unicode_cleaner(k, v)
                    for k, v in record.items()
                    if k in vespa_record["fields"].keys()
                }
            )

            output_vespa.append(vespa_record)

        json.dump(output_vespa, out)


if __name__ == "__main__":
    main()
