""" This file contains code needed to prepare the scraped Epicurious recipe 
JSON to convert to a database that can be used for cosine similarity analysis.
"""

# Import necessary libraries
import json
import csv
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load stopwords and prepare lemmatizer
stopwords_loc = "../write_data/food_stopwords.csv"
with open(stopwords_loc, "r") as myfile:
    reader = csv.reader(myfile)
    food_stopwords = [col for row in reader for col in row]

stopwords_list = stopwords.words("english") + list(string.punctuation) + food_stopwords

lemmatizer = WordNetLemmatizer()


# Define functions
def cuisine_namer(text):
    """This function converts redundant and/or rare categories into more common 
  ones/umbrella ones.
  
  In the future, there's a hope that this renaming mechanism will not have 
  under sampled cuisine tags.
  """
    if text == "Central American/Caribbean":
        return "Caribbean"
    elif text == "Jewish":
        return "Kosher"
    elif text == "Eastern European/Russian":
        return "Eastern European"
    elif text in ["Spanish/Portuguese", "Greek"]:
        return "Mediterranean"
    elif text == "Central/South American":
        return "Latin American"
    elif text == "Sushi":
        return "Japanese"
    elif text == "Southern Italian":
        return "Italian"
    elif text in ["Southern", "Tex-Mex"]:
        return "American"
    elif text in ["Southeast Asian", "Korean"]:
        return "Asian"
    else:
        return text


filename = "../raw_data/recipes-en-201706/epicurious-recipes_m2.json"
with open(filename, "r") as f:
    datastore = json.load(f)


def load_data(filepath, test_size=0.1, random_state=10):
    """ This function uses a filepath, test_size, and random_state
    to load the Epicurious JSON into a dataframe and then split into 
    train/test sets."""
    with open(filepath, "r") as f:
        datastore = json.load(f)
    datastore_df = pd.DataFrame(datastore)
    X_train, X_test = train_test_split(
        datastore_df, test_size=test_size, random_state=random_state
    )
    return X_train, X_test


def prep_data(X):
    """ This function takes a dataframe X, drops columns that will not be used,
    expands the hierarchical column into the dataframe, renames the columns
    to be more human-readable, and drops one column created during dataframe
    expansion"""
    X.drop(
        [
            "pubDate",
            "author",
            "type",
            "aggregateRating",
            "reviewsCount",
            "willMakeAgainPct",
            "dateCrawled",
        ],
        axis=1,
        inplace=True,
    )

    concat = pd.concat([X.drop(["tag"], axis=1), X["tag"].apply(pd.Series)], axis=1)
    concat.drop(
        [
            0,
            "photosBadgeAltText",
            "photosBadgeFileName",
            "photosBadgeID",
            "photosBadgeRelatedUri",
        ],
        axis=1,
        inplace=True,
    )

    cols = [
        "id",
        "description",
        "title",
        "url",
        "photo_data",
        "ingredients",
        "steps",
        "category",
        "name",
        "remove",
    ]

    concat.columns = cols

    concat.drop("remove", axis=1, inplace=True)

    cuisine_only = concat[concat["category"] == "cuisine"]
    cuisine_only.dropna(axis=0, inplace=True)
    cuisine_only["imputed_label"] = cuisine_only["name"].apply(cuisine_namer)

    return cuisine_only


def fit_transform_tfidf_matrix(X_df, stopwords_list):
    tfidf = TfidfVectorizer(
        stop_words=stopwords_list,
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        preprocessor=lemmatizer.lemmatize,
    )

    temp = X_df["ingredients"].apply(" ".join).str.lower()
    tfidf.fit(temp)
    response = tfidf.transform(temp)
    print(response.shape)
    word_matrix = pd.DataFrame(
        response.toarray(), columns=tfidf.get_feature_names(), index=X_df.index
    )

    return tfidf, word_matrix


def transform_tfidf(tfidf, recipe):
    response = tfidf.transform(recipe["ingredients"])

    transformed_recipe = pd.DataFrame(
        response.toarray(), columns=tfidf.get_feature_names(), index=recipe.index
    )
    return transformed_recipe


def transform_from_test_tfidf(tfidf, df, idx):
    recipe = [" ".join(df.iloc[idx]["ingredients"])]
    response = tfidf.transform(recipe)
    transformed_recipe = pd.DataFrame(
        response.toarray(), columns=tfidf.get_feature_names()
    )
    return transformed_recipe


def filter_out_cuisine(ingred_word_matrix, X_df, cuisine_name, tfidf):
    combo = pd.concat([ingred_word_matrix, X_df["imputed_label"]], axis=1)
    filtered_ingred_word_matrix = combo[combo["imputed_label"] != cuisine_name].drop(
        "imputed_label", axis=1
    )
    return filtered_ingred_word_matrix


def find_closest_recipes(filtered_ingred_word_matrix, recipe_tfidf, X_df):
    search_vec = np.array(recipe_tfidf).reshape(1, -1)
    res_cos_sim = cosine_similarity(filtered_ingred_word_matrix, search_vec)
    top_five = np.argsort(res_cos_sim.flatten())[-5:][::-1]
    proximity = res_cos_sim[top_five]
    recipe_ids = [filtered_ingred_word_matrix.iloc[idx].name for idx in top_five]
    suggest_df = X_df.loc[recipe_ids]
    return suggest_df, proximity


# Create the dataframe
X_train, X_test = load_data(filename)

with open("joblib/test_subset.joblib", "wb") as fo:
  joblib.dump(X_test, fo, compress=True)

prepped = prep_data(X_train)
with open("joblib/recipe_dataframe.joblib", "wb") as fo:
  joblib.dump(prepped, fo, compress=True)

# Create the ingredients TF-IDF matrix
ingred_tfidf, ingred_word_matrix = fit_transform_tfidf_matrix(prepped, stopwords_list)
with open("joblib/recipe_tfidf.joblib", "wb") as fo:
  joblib.dump(ingred_tfidf, fo, compress=True)

with open("joblib/recipe_word_matrix.joblib", "wb") as fo:
  joblib.dump(ingred_word_matrix, fo, compress=True)
