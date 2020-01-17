""" This file contains code needed to prepare the scraped Epicurious recipe 
JSON to convert to a database that can be used for cosine similarity analysis.

For demonstration purposes, this file converts the ingredients via the 
naive One Hot Encode method.
"""

# Import necessary libraries
import json
import csv
import re
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Load stopwords and prepare lemmatizer
stopwords_loc = "../food_stopwords.csv"
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


filename = "../secrets/recipes-en-201706/epicurious-recipes_m2.json"
with open(filename, "r") as f:
    datastore = json.load(f)
    f.close()


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
          'prepSteps'
      ],
      axis=1,
      inplace=True,
  )

  X.rename({'url': 'recipe_url'}, axis=1, inplace=True)     
  
  concat = pd.concat([X.drop(["tag"], axis=1), X["tag"].apply(pd.Series)], axis=1)
  concat.drop(
      [
          0,
          "photosBadgeAltText",
          "photosBadgeFileName",
          "photosBadgeID",
          "photosBadgeRelatedUri",
          "url"
      ],
      axis=1,
      inplace=True,
  )
  cuisine_only = concat[concat["category"] == "cuisine"]
  cuisine_only.dropna(axis=0, inplace=True)
  cuisine_only["imputed_label"] = cuisine_only["name"].apply(cuisine_namer)
  cuisine_only.drop('name', axis=1, inplace=True)
  return cuisine_only

def lemmatize_training_recipes(ingredients, stopwords_list):
  list_ingreds = ingredients.tolist()
  no_dash_ingreds = [ingred.replace('-', ' ') for ingred in list_ingreds]
  no_splat_ingreds = [ingred.replace('*', ' ') for ingred in no_dash_ingreds]
  recipe_ingreds = [ingred.replace('/', ' ') for ingred in no_splat_ingreds]
  token_recipes = [word_tokenize(ingred) for ingred in recipe_ingreds]

  unique_ingreds = set()
  lemmatized_recipes = []

  for recipe in token_recipes:
    lemmatized_recipe = []
    for token in recipe:
        try:
          float(token)
          continue
        except:
          if token in stopwords_list:
            continue
          else:
            unique_ingreds.add(lemmatizer.lemmatize(token)) 
            lemmatized_recipe.append(lemmatizer.lemmatize(token))

    lemmatized_recipes.append(lemmatized_recipe)

  return lemmatized_recipes, unique_ingreds

def fit_transform_ohe_matrix(X_df, stopwords_list):
  ohe = CountVectorizer(
        stop_words=stopwords_list,
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        preprocessor=lemmatizer.lemmatize,
    )
  ingreds = X_df["ingredients"].apply(" ".join).str.lower()
  ohe.fit(ingreds)
  response = ohe.transform(ingreds)
  ohe_matrix = pd.DataFrame(response.toarray(), 
                              columns=ohe.get_feature_names(), 
                              index=X_df.index
                            )
  return ohe, ohe_matrix
  

def transform_ohe(ohe, recipe):
  ingreds = recipe['ingredients'].apply(" ".join).str.lower()
  response = ohe.transform(ingreds)

  ohe_transformed_recipe = pd.DataFrame(
                                        response.toarray(),
                                        columns=ohe.get_feature_names(),
                                        index=recipe.index
                                        )
  print(ohe_transformed_recipe.to_numpy().where(ohe_transformed_recipe.to_numpy() != (0.0 or 1.0), 'not 1 or 0', '1 or 0'))
  return ohe_transformed_recipe


def transform_from_test_ohe(ohe, df, idx):
  recipe = df['ingredients'].iloc[idx].apply(' '.join).str.lower()
  response = ohe.transform(recipe)
  ohe_transformed_test_recipe = pd.DataFrame(
                                            response.toarray(), 
                                            columns=ohe.get_feature_names()
                                            )

  return ohe_transformed_test_recipe


def filter_out_cuisine(ingred_word_matrix, X_df, cuisine_name, ohe):
  combo = pd.concat([ingred_word_matrix, X_df["imputed_label"]], axis=1)
  filtered_ingred_word_matrix = combo[combo["imputed_label"] != cuisine_name].drop(
      "imputed_label", axis=1
  )
  return filtered_ingred_word_matrix


def find_closest_recipes(filtered_ingred_word_matrix, recipe_ohe_transform, X_df):
  search_vec = np.array(recipe_ohe_transform).reshape(1, -1)
  res_cos_sim = cosine_similarity(filtered_ingred_word_matrix, search_vec)
  top_five = np.argsort(res_cos_sim.flatten())[-5:][::-1]
  proximity = res_cos_sim[top_five]
  recipe_ids = [filtered_ingred_word_matrix.iloc[idx].name for idx in top_five]
  suggest_df = X_df.loc[recipe_ids]

  return suggest_df, proximity


# Create the dataframe
X_train, X_test = load_data(filename)

with open("../joblib/ohe_test_subset.joblib", "wb") as fo:
  joblib.dump(X_test, fo, compress=True)

prepped = prep_data(X_train)
with open("../joblib/ohe_recipe_dataframe.joblib", "wb") as fo:
  joblib.dump(prepped, fo, compress=True)

# Create the ingredients OHE matrix
ingred_ohe, ingred_word_matrix = fit_transform_ohe_matrix(prepped, stopwords_list)
with open("../joblib/recipe_ohe.joblib", "wb") as fo:
  joblib.dump(ingred_ohe, fo, compress=True)

with open("../joblib/recipe_word_matrix_ohe.joblib", "wb") as fo:
  joblib.dump(ingred_word_matrix, fo, compress=True)
