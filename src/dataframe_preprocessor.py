""" This script is intended to take in a pandas dataframe from read_json for the scraped Epicurious data and preprocess the dataframe to prepare it for natural language processing down the line. 
"""

import pandas as pd
from typing import Dict, Text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """This function takes in a pandas DataFrame from pd.read_json and performs some preprocessing by unpacking the nested dictionaries and creating new columns with the simplified structures. It will then drop the original columns that would no longer be needed.

    Args:
        pd.DataFrame

    Returns:
        pd.DataFrame
    """

    def link_maker(recipe_link: Text) -> Text:
        """This function takes in the incomplete recipe link from the dataframe and returns the complete one."""
        full_link = f"https://www.epicurious.com{recipe_link}"
        return full_link

    def null_filler(to_check: Dict[Text, Text], key_target: Text) -> Text:
        """This function takes in a dictionary that is currently fed in with a lambda function and then performs column specific preprocessing.

        Args:
            to_check: dict
            key_target: str

        Returns:
            str
        """

        # Only look in the following keys, if the input isn't one of these, it should be recognized as an improper key
        valid_keys = ["name", "filename", "credit"]

        # This dictionary converts the input keys into substrings that can be used in f-strings to fill in missing values in the record
        translation_keys = {
            "name": "Cuisine",
            "filename": "Photo",
            "credit": "Photo Credit",
        }

        if key_target not in valid_keys:
            # this logic makes sure we are only looking at valid keys
            return (
                "Improper key target: can only pick from 'name', 'filename', 'credit'."
            )

        else:
            if pd.isna(to_check):  # type:ignore
                # this logic checks to see if the dictionary exists at all. if so, return Missing
                return f"Missing {translation_keys[key_target]}"
            else:
                if key_target == "name" and (to_check["category"] != "cuisine"):
                    # This logic checks for the cuisine, if the cuisine is not there (and instead has 'ingredient', 'type', 'item', 'equipment', 'meal'), mark as missing
                    return f"Missing {translation_keys[key_target]}"
                else:
                    # Otherwise, there should be no issue with returning
                    return to_check[key_target]

    # Dive into the tag column and extract the cuisine label. Put into new column or fills with "missing data"
    df["cuisine_name"] = df["tag"].apply(
        lambda x: null_filler(to_check=x, key_target="name")
    )  # type:ignore

    # this lambda function goes into the photo data column and extracts just the filename from the dictionary
    df["photo_filename"] = df["photoData"].apply(
        lambda x: null_filler(to_check=x, key_target="filename")
    )  # type:ignore

    # This lambda function goes into the photo data column and extracts just the photo credit from the dictionary
    df["photo_credit"] = df["photoData"].apply(
        lambda x: null_filler(to_check=x, key_target="credit")
    )  # type:ignore

    # for the above, maybe they can be refactored to one function where the arguments are a column name, dictionary key name, the substring return

    # this lambda funciton goes into the author column and extract the author name or fills iwth "missing data"
    df["author_name"] = df["author"].apply(
        lambda x: x[0]["name"] if x else "Missing Author Name"
    )  # type:ignore

    # This function takes in the given pubDate column and creates a new column with the pubDate values converted to datetime objects
    df["date_published"] = pd.to_datetime(
        df["pubDate"], infer_datetime_format=True
    )  # type:ignore

    # this function takes in the given url column and prepends the full epicurious URL base
    df["recipe_url"] = df["url"].apply(link_maker)  # type:ignore

    # drop some original columns to clean up the dataframe
    df.drop(
        labels=["tag", "photoData", "author", "type", "dateCrawled", "pubDate", "url"],
        axis=1,
        inplace=True,
    )

    return df
