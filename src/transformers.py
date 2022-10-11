"""
This script groups together the NLP steps and creation/modification of the pandas dataframe containing all of the recipe information and then the TF-IDF transformed matrices concatenated on. It is intended to feed into the plotter script.
"""

# Import libraries
from datetime import datetime
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib.text as mlt
import numpy as np
from openTSNE.sklearn import TSNE
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import spacy
import en_core_web_sm
from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm
from typing import Any
import umap

# Other custom functions in same folder
import dataframe_preprocessor as dfpp
import nlp_processor as nlp_proc


def prepare_dataframe(
    data_path: Text = "../../data/recipes-en-201706/epicurious-recipes_m2.json",
) -> pd.DataFrame:
    """
    This function uses the preprocess script to prepare the dataframe

    Args:
        data_path: string representing location of data

    Returns:
        df: processed pandas DataFrame
    """
    source_df = pd.read_json(path_or_buf=data_path)

    preprocessed_df = dfpp.preprocess_dataframe(df=source_df)

    return preprocessed_df


def prepare_nlp(
    stopwords_path: Text = "../../food_stopwords.csv",
    pretrained_parameter: Text = "en_core_web_sm",
) -> tuple[Language, list[float | _str | LiteralString]]:
    """
    This function prepares the items needed to do the CountVectorization. It will perform text processing with spaCy and generate the custom stopwords set

    Args:
        stopwords_path: string representing location of custom stopwords
        pretrained_parameters: string representing pretrained spaCy model

    Returns:
        NLP_Processor: custom class/object based on spaCy

    """
    nlp = spacy.load(pretrained_parameter)

    total_stopwords = {x for x in pd.read_csv(stopwords_path)}
    cooking_specific_stopwords = {
        "red",
        "green",
        "black",
        "yellow",
        "white",
        "inch",
        "mince",
        "chop",
        "fry",
        "trim",
        "flat",
        "beat",
        "brown",
        "golden",
        "balsamic",
        "halve",
        "blue",
        "divide",
        "trim",
        "unbleache",
        "granulate",
        "Frank",
        "alternative",
        "american",
        "annie",
        "asian",
        "balance",
        "band",
        "barrel",
        "bay",
        "bayou",
        "beam",
        "beard",
        "bell",
        "betty",
        "bird",
        "blast",
        "bob",
        "bone",
        "breyers",
        "calore",
        "carb",
        "card",
        "chachere",
        "change",
        "circle",
        "coffee",
        "coil",
        "country",
        "cow",
        "crack",
        "cracker",
        "crocker",
        "crystal",
        "dean",
        "degree",
        "deluxe",
        "direction",
        "duncan",
        "earth",
        "eggland",
        "ener",
        "envelope",
        "eye",
        "fantastic",
        "far",
        "fat",
        "feather",
        "flake",
        "foot",
        "fourth",
        "frank",
        "french",
        "fusion",
        "genoa",
        "genovese",
        "germain",
        "giada",
        "gold",
        "granule",
        "greek",
        "hamburger",
        "helper",
        "herbe",
        "hines",
        "hodgson",
        "hunt",
        "instruction",
        "interval",
        "italianstyle",
        "jim",
        "jimmy",
        "kellogg",
        "lagrille",
        "lake",
        "land",
        "laurentiis",
        "lawry",
        "lipton",
        "litre",
        "ll",
        "maid",
        "malt",
        "mate",
        "mayer",
        "meal",
        "medal",
        "medallion",
        "member",
        "mexicanstyle",
        "monte",
        "mori",
        "nest",
        "nu",
        "oounce",
        "oscar",
        "ox",
        "paso",
        "pasta",
        "patty",
        "petal",
        "pinche",
        "preserve",
        "quartere",
        "ranch",
        "ranchstyle",
        "rasher",
        "redhot",
        "resemble",
        "rice",
        "ro",
        "roni",
        "scissor",
        "scrap",
        "secret",
        "semicircle",
        "shard",
        "shear",
        "sixth",
        "sliver",
        "smucker",
        "snicker",
        "source",
        "spot",
        "state",
        "strand",
        "sun",
        "supreme",
        "tablepoon",
        "tail",
        "target",
        "tm",
        "tong",
        "toothpick",
        "triangle",
        "trimming",
        "tweezer",
        "valley",
        "vay",
        "wise",
        "wishbone",
        "wrapper",
        "yoplait",
        "ziploc",
    }

    total_stopwords = total_stopwords.union(STOP_WORDS)
    total_stopwords = total_stopwords.union(cooking_specific_stopwords)

    total_stopwords_list = list(total_stopwords)

    return nlp, total_stopwords_list


def text_handling_transformer_pipeline(
    preprocessed_df: pd.DataFrame, nlp_processor: Any, custom_stopwords: List
) -> Pipeline:
    """
    This function takes the preprocessed dataframe, custome NLP Processor, and custom stopwords to perform sklearn pipeline to result in transformed dataframe
    """
    ingredient_megalist = [
        ingred
        for recipe in preprocessed_df["ingredients"].tolist()
        for ingred in recipe
    ]

    pipeline = Pipeline(
        [
            ("countvectorizer", CountVectorizer()),
            ("tfwhydf", TfidfTransformer()),
            ("tsvd", TruncatedSVD()),
        ],
        verbose=True
    )

    parameters = {
        "countvectorizer__strip_accents": "unicode",
        "countvectorizer__lowercase": True,
        "countvectorizer__preprocessor": custom_nlp_proc.custom_preprocessor,
        "countvectorizer__tokenizer": custom_nlp_proc.custom_lemmatizer,
        "countvectorizer__stop_words": flushtrated_list,
        "countvectorizer__token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
        "countvectorizer__ngram_range": (1, 4),
        "countvectorizer__min_df": 10,
        "tsvd__n_components": 100,
        "tsvd__n_iter": 15,
        "tsvd__random_state": 268,
    }

    return Pipeline


def concat_matrices_to_df(
    df: pd.DataFrame, vectorized_ingred_matrix: scipy.sparse.csr_matrix, cv: Any
) -> pd.DataFrame:
    """
    This function takes in a dataframe and concats the matrix generated by either CountVectorizer or TFIDF-Transformer onto the records so that the recipes can be used for classification purposes.

    Args:
        df: preprocessed dataframe from preprocess_dataframe
        vectorized_ingred_matrix: sparse csr matrix created from doing fit_transform on the recipe_megalist
        cv: sklearn CountVectorizer object

    Returns:
        A pandas dataframe with the vectorized_ingred_matrix appended as columns to df
    """
    repo_tfidf_df = pd.DataFrame(
        vectorized_ingred_matrix.toarray(),
        columns=cv.get_feature_names_out(),
        index=df.index,
    )
    return pd.concat([df, repo_tfidf_df], axis=1)


def dataframe_filter(df_with_cv: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes in the recipe dataframe that has been concatenated with ingredient vectors and removes the unneeded columns to use further down the line

    Args: 
        df_with_cv: Pandas DataFrame coming out of concat_matrices_to_df()

    Returns:
        reduced_df: Pandas DataFrame
    """
    
    reduced_df = df_with_cv.drop(['dek', 'hed', 'aggregateRating', 'ingredients', 'prepSteps',
       'reviewsCount', 'willMakeAgainPct', 'photo_filename',
       'photo_credit', 'author_name', 'date_published', 'recipe_url'], axis=1)

    return reduced_df


def classifying_pipeline(reduced_df: pd.DataFrame, random_state: int = 240): 
    """
    This function takes in the reduced_df, performs train/test split, performs dimension reduction via truncatedSVD and tSNE

    Args: 
        reduced_df: pd.DataFrame

    Returns
        transformed model
    """
    pipeline = Pipeline(
        [
           ("tsvd", TruncatedSVD()),
           ("tsne", TSNE()),
           ("kmeans", KMeans())
        ],
        verbose=True
    )

    parameters = {
        "tsvd__n_components": 100,
        "tsvd__n_iter": 15,
        "tsvd__random_state": 268,
        "tsne__n_components": 2,
        "tsne__learning_rate": "auto",
        "tsne__perplexity": 500,
        "tsne__random_state": 144,
        "tnse__n_jobs": -1,
        "kmeans__n_clusters": 12,
        "kmeans__random_state": 30,
    }

    return Pipeline
# filter and clean dataframe
# CountVectorizer, TF-IDF
# TruncatedSVD, tSNE
