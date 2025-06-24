"""
This script groups together the NLP steps, creation/modification of the pandas dataframe containing all of the recipe information, and then concatenates on the TF-IDF transformed ingedient matrix. It is intended to feed into the plotter script.
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
from typing import Any, Dict, List, Text, Tuple
import umap

# Other custom functions in same folder
import src.dataframe_preprocessor as dfpp
import src.nlp_processor as nlp_proc


def prepare_dataframe(
    data_path: Text = "../../data/recipes-en-201706/epicurious-recipes_m2.json",
) -> pd.DataFrame:
    """
    This function uses the preprocess script to prepare the dataframe

    Args:
        data_path: string representing location of data

    Returns:
        preprocessed_df: processed pandas DataFrame
    """
    source_df = pd.read_json(path_or_buf=data_path)

    preprocessed_df = dfpp.preprocess_dataframe(df=source_df)

    return preprocessed_df


def prepare_nlp(
    stopwords_path: Text = "../../food_stopwords.csv",
    pretrained_parameter: Text = "en_core_web_sm",
) -> Tuple[Any, List[Text]]:
    """
    This function prepares the items needed to do the CountVectorization step via sklearn. It will perform text processing with spaCy, imports the default stopwords set and combines it with the subject matter expert determined list of stopwords, generates the custom stopwords set, and returns the NLP processor for use down the line.

    Args:
        stopwords_path: string representing location of custom stopwords in a csv
        pretrained_parameters: string representing pretrained spaCy model to import

    Returns:
        NLP_Processor: custom class/object based on spaCy. This can be pickled/joblib stored in case of recall

    """
    nlp = nlp_proc.NLP_Processor(pretrained_parameter)

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
    preprocessed_df: pd.DataFrame, custom_nlp: Any, custom_stopwords: List[Text]
) -> Tuple["pd.DataFrame", "Pipeline"]:
    """
    This function takes the preprocessed dataframe, custom NLP Processor, and custom stopwords and uses sklearn pipeline to fit and transform via CountVectorizer and TF-IDF to return a transformed dataframe and the fitted pipeline for use later

    Args:
        preprocessed_df: pandas DataFrame from prepare_dataframe above
        custom_stopwords: list of words

    Returns:
        Tuple:
            tht_transformed: pandas DataFrame with CV/TFIDF done
            tnt_pipe: sklearn pipeline to be used if needed (say with a new recipe)
    """
    ingredient_megalist = [
        ingred
        for recipe in preprocessed_df["ingredients"].tolist()
        for ingred in recipe
    ]

    # transformers = Pipeline(
    #     steps=[("countvectorizer", CountVectorizer()), ("tfwhydf", TfidfTransformer())],
    #     verbose=True,
    # )

    # parameters = {
    #     "steps__countvectorizer": {
    #         "strip_accents": "unicode",
    #         "lowercase": True,
    #         "preprocessor": custom_nlp.custom_preprocessor,
    #         "tokenizer": custom_nlp.custom_lemmatizer,
    #         "stop_words": custom_stopwords,
    #         "token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
    #         "ngram_range": (1, 4),
    #         "min_df": 10
    #     }

    cv_params = {
        "strip_accents": "unicode",
        "lowercase": True,
        "preprocessor": custom_nlp.custom_preprocessor,
        "tokenizer": custom_nlp.custom_lemmatizer,
        "stop_words": custom_stopwords,
        "token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
        "ngram_range": (1, 4),
        "min_df": 10,
    }

    tht_pipe = Pipeline(
        steps=[
            ("countvectorizer", CountVectorizer(**cv_params)),
            ("tfwhydf", TfidfTransformer()),
        ],
        verbose=True,
    )
    # "steps__countvectorizer__strip_accents": "unicode",
    # "steps__countvectorizer__lowercase": True,
    # "steps__countvectorizer__preprocessor": custom_nlp.custom_preprocessor,
    # "steps__countvectorizer__tokenizer": custom_nlp.custom_lemmatizer,
    # "steps__countvectorizer__stop_words": custom_stopwords,
    # "steps__countvectorizer__token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
    # "steps__countvectorizer__ngram_range": (1, 4),
    # "steps__countvectorizer__min_df": 10,
    # }

    # tht_pipe = Pipeline(transformers)
    # tht_pipe.set_params(**parameters)

    tht_transformed = tht_pipe.fit(ingredient_megalist)

    temp = preprocessed_df["ingredients"].apply(" ".join).str.lower()
    tht_transformed = tht_pipe.transform(temp)

    return tht_transformed, tht_pipe


def concat_matrices_to_df(
    preprocessed_df: pd.DataFrame,
    tht_transformed_matrix: Any,  # scipy.sparse.csr_matrix is actual return, but not wanting to import scipy just for type hinting
    tht_pipe: Pipeline,
) -> pd.DataFrame:
    """
    This function takes in a dataframe and concatenates on the matrix generated by either CountVectorizer or TFIDF-Transformer (realistically, the output from the pipeline above) onto the records so that the recipes can be used for classification purposes.

    Args:
        preprocessed_df: preprocessed dataframe from preprocess_dataframe
        tht_transformed_matrix: sparse csr matrix created from doing fit_transform on the recipe_megalist (usually vectorized_ingred_matrix from text_handling_transformer_pipeline above)
        tht_pipe: sklearn Pipeline object from text_handling_transformer_pipeline above, but really a CountVectorizer object or TF-IDF Transformer object

    Returns:
        A pandas dataframe with the vectorized_ingred_matrix appended as columns to preprocessed_df
    """
    repo_tfidf_df = pd.DataFrame(
        tht_transformed_matrix.toarray(),
        columns=tht_pipe.get_feature_names_out(),
        index=preprocessed_df.index,
    )
    return pd.concat([preprocessed_df, repo_tfidf_df], axis=1)


def dataframe_filter(df_with_cv: pd.DataFrame, random_state: int = 240) -> pd.DataFrame:
    """
    This function takes in the recipe dataframe that has been concatenated with ingredient vectors and removes the unneeded columns and recipes missing cuisine labels to use further down the line

    Args:
        df_with_cv: Pandas DataFrame coming out of concat_matrices_to_df()

    Returns:
        reduced_df2: Cleaned Pandas DataFrame
    """

    reduced_df = df_with_cv.drop(
        [
            "dek",
            "hed",
            "aggregateRating",
            "ingredients",
            "prepSteps",
            "reviewsCount",
            "willMakeAgainPct",
            "photo_filename",
            "photo_credit",
            "author_name",
            "date_published",
            "recipe_url",
        ],
        axis=1,
    )

    reduced_df2 = reduced_df[reduced_df["cuisine_name"] != "Missing Cuisine"]

    return reduced_df2


def find_important_ingredients(
    recipes_post_cv_df: pd.DataFrame, n_most: int = 5
) -> pd.DataFrame:
    """
    This function takes in the pandas DataFrame containing the processed recipes concatenated with their CountVectorizer/TF-IDF transformed sparse ingredient matrices and returns a new pandas DataFrame with recipe ID as an index and top n_most ingredients as a column.

    Args:
        recipes_post_cv_df: pd.DataFrame, concatenated recipe df from concat_matrices_to_df
        n_most: int, number of ingredients to include

    Returns:
        pd.DataFrame
    """
    sparse = recipes_post_cv_df.drop(
        [
            "dek",
            "hed",
            "aggregateRating",
            "ingredients",
            "prepSteps",
            "reviewsCount",
            "willMakeAgainPct",
            "cuisine_name",
            "photo_filename",
            "photo_credit",
            "author_name",
            "date_published",
            "recipe_url",
        ],
        axis=1,
    )

    important_ingreds_indices = sparse.apply(
        lambda x: x.argsort()[-5:].values.tolist(), axis=1
    )

    important_ingredients_df = pd.DataFrame(
        data={
            "important_ingredients": [
                sparse.loc[idx].iloc[important_ingreds_indices.loc[idx]].index.tolist()
                for idx in sparse.index
            ]
        },
        index=important_ingreds_indices.index,
    )

    return important_ingredients_df


def classifying_pipeline(reduced_df: pd.DataFrame, random_state: int = 240):
    """
    This function takes in the reduced_df, performs train/test split, performs dimension reduction via truncatedSVD and tSNE

    Args:
        reduced_df: pd.DataFrame

    Returns
        Tuple[
            tsne_transformed: np.Array of pipeline transformed training data,
            X_train: pd.DataFrame of training data,
            y_train: pd.Series of training labels,
            tsne_transformed_test: np.Array of pipeline transformed test data,
            X_test: pd.DataFrame of test data
            y_test: pd.Series of test labels,
            clf_pipe: Pipeline of transformers
            ]
    """

    y = reduced_df["cuisine_name"]
    X = reduced_df.drop(["cuisine_name"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=240, stratify=y
    )

    tsvd_params = {
        "n_components": 100,
        "n_iter": 15,
        "random_state": 268,
    }

    tsne_params = {
        "n_components": 2,
        "learning_rate": "auto",
        "perplexity": 500,
        "random_state": 144,
        "n_jobs": -1,
    }

    clf_pipe = Pipeline(
        steps=[("tsvd", TruncatedSVD(**tsvd_params)), ("tsne", TSNE(**tsne_params))],
        verbose=True,
    )

    # the below parameters did not work for set_params
    # parameters = {
    #     "steps__tsvd__n_components": 100,
    #     "steps__tsvd__n_iter": 15,
    #     "steps__tsvd__random_state": 268,
    #     "steps__tsne__n_components": 2,
    #     "steps__tsne__learning_rate": "auto",
    #     "steps__tsne__perplexity": 500,
    #     "steps__tsne__random_state": 144,
    #     "steps__tnse__n_jobs": -1,
    # }

    # clf_pipe = Pipeline(reduced_dim_labeler)
    # clf_pipe.set_params(**parameters)

    tsne_transformed = clf_pipe.fit_transform(X_train)

    tsne_transformed_test = clf_pipe.transform(X_test)

    return (
        tsne_transformed,
        X_train,
        y_train,
        tsne_transformed_test,
        X_test,
        y_test,
        clf_pipe,
    )


def attach_important_ingreds(
    tsne_transformed_np: np.ndarray,
    X: pd.DataFrame,
    y: pd.DataFrame,
    important_ingredients_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    This function takes in the tSNE pipeline pipeline transformed DataFrame and joins it with the important ingredient dataframe to display the n-most important ingredients in the bokeh plot.

    Args:
        tsne_transformed_np: np.ndarray from classifying_pipeline above (either train or test)
        X: pd.DataFrame from classifying_pipeline above (either train or test)
        important_ingredients_df: pd.DataFrame from find_imporant_ingredients above

    Returns:
        pd.DataFrame combining both
    """
    tsne_transformed_df = pd.DataFrame(
        data=tsne_transformed_np, index=X.index, columns=["X", "Y"]
    )
    tsne_transformed_df["cuisine_name"] = y

    tsne_transformed_df2 = tsne_transformed_df.join(
        important_ingredients_df, how="inner"
    )
    return tsne_transformed_df2
