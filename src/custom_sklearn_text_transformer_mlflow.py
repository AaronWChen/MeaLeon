from itertools import tee, islice
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
import stanza
import tqdm


class CustomSKLearnAnalyzer:
    """
    This class handles using Stanza with a custom analyzer inside sklearn
    """

    def __init__(self, stanza_lang_str="en"):
        """
        Constructor method. Initializes the model with a Stanza libary language
        type. The default is "en" for English, later on, can think adding
        functionality to download the pretrained model/embeddings
        """
        self.stanza_lang_str = stanza_lang_str

    def prepare_stanza_pipeline(
        self,
        depparse_batch_size=50,
        depparse_min_length_to_batch_separately=50,
        verbose=True,
        use_gpu=False,
        batch_size=100,
    ):
        """
        Method to simply construction of Stanza Pipeline for usage in the sklearn custom analyzer

        Args:
            Follow creation of stanza pipeline (link to their docs)

            self.stanza_lang_str:
                str for pretrained Stanza embeddings to use in the pipeline (from init)

            depparse_batch_size:
                int for batch size for processing, default is 50

            depparse_min_length_to_batch_separately:
                int for minimum string length to batch, default is 50

            verbose:
                boolean for information for readouts during processing, default is True

            use_gpu:
                boolean for using GPU for stanza, default is False,
                set to True when on cloud/not on streaming computer

            batch_size:
                int for batch sizing, default is 100

        Returns:
            nlp:
                stanza pipeline
        """

        # Perhaps down the road, this should be stored as an MLflow Artifact to be downloaded
        # Or should this be part of the Container building at start up? If so, how would those get logged? Just as artifacts?
        stanza.download(self.stanza_lang_str)

        nlp = stanza.Pipeline(
            self.stanza_lang_str,
            depparse_batch_size=depparse_batch_size,
            depparse_min_length_to_batch_separately=depparse_min_length_to_batch_separately,
            verbose=verbose,
            use_gpu=use_gpu,
            batch_size=batch_size,
        )

        return nlp

    def fit_transform(
        self,
        input_data,
        stanza_pipeline,
        strip_accents="unicode",
        lowercase=True,
        min_ngram_length=1,
        max_ngram_length=4,
        min_df=3,
        sklearn_type="OneHotEncode",
    ):
        """
        Method to simplify construction of custom sklearn text processor.

        Follows construction of standard CountVectorizer/TFIDFVectorizer

        Args:
            Follows sklearn CountVectorizer construction with some changes:

            input_data:
                pd.Series to be transformed. Each element in the series should be list of strings

            stanza_pipeline:
                stanza.pipeline from prepare_stanza_pipeline

            min_ngram_length:
                setting for minimum number in ngram vectoriazation,
                used with custom analyzer
                default of 1

            max_ngram_length:
                setting for maximum number in ngram vectoriazation,
                used with custom analyzer
                default of 4

            sklearn_type:
                Setting for OneHotEncode, Regular CountVectorization, or TFIDFVectorization
                default for OneHotEncode, choose between "OneHotEncode", "CountVectorizer", "TFIDF"

        Returns:
            sklearn_transformer:
                sklearn text transformer for usage later/in MLflow models

            transformed_text:
                pd.DataFrame that combines the vectorized text with the original dataframe
        """

        sklearn_transformer_params = {
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "min_df": min_df,
            "analyzer": CustomSKLearnAnalyzer().stanza_analyzer(
                stanza_pipeline=stanza_pipeline,
                min_ngram_length=min_ngram_length,
                max_ngram_length=max_ngram_length,
            ),
            "sklearn_type": sklearn_type,
        }

        sklearn_transformer = CustomSKLearnAnalyzer().fit(**sklearn_transformer_params)

        transformed_recipe = CustomSKLearnAnalyzer().transform(
            sklearn_transformer=sklearn_transformer,
            input_data=input_data,
        )

        return sklearn_transformer, transformed_recipe

    def transform(
        self,
        sklearn_transformer,
        input_data,
        stanza_pipeline,
        strip_accents="unicode",
        lowercase=True,
        min_ngram_length=1,
        max_ngram_length=4,
        min_df=3,
        sklearn_type="OneHotEncode",
    ):
        """
        Method to simplify construction of custom sklearn text processor.

        Follows construction of standard CountVectorizer/TFIDFVectorizer

        Args:
            Follows sklearn CountVectorizer construction with some changes:

            input_data:
                pd.Series to be transformed. Each element in the series should be list of strings

            stanza_pipeline:
                stanza.pipeline from prepare_stanza_pipeline

            min_ngram_length:
                setting for minimum number in ngram vectoriazation,
                used with custom analyzer
                default of 1

            max_ngram_length:
                setting for maximum number in ngram vectoriazation,
                used with custom analyzer
                default of 4

            sklearn_type:
                Setting for OneHotEncode, Regular CountVectorization, or TFIDFVectorization
                default for OneHotEncode, choose between "OneHotEncode", "CountVectorizer", "TFIDF"

        Returns:
            sklearn_transformer:
                sklearn text transformer for usage later/in MLflow models

            transformed_text:
                pd.DataFrame that combines the vectorized text with the original dataframe
        """

        response = sklearn_transformer.transform(input_data)

        transformed_recipe = pd.DataFrame(
            response.toarray(),
            columns=sklearn_transformer.get_feature_names_out(),
            index=input_data.index,
        )

        return transformed_recipe

    def fit(
        self,
        input_data,
        stanza_pipeline,
        strip_accents="unicode",
        lowercase=True,
        min_ngram_length=1,
        max_ngram_length=4,
        min_df=3,
        sklearn_type="OneHotEncode",
    ):
        """
        Method to simplify construction of custom sklearn text processor.

        Follows construction of standard CountVectorizer/TFIDFVectorizer

        Args:
            Follows sklearn CountVectorizer construction with some changes:

            input_data:
                pd.Series to be transformed. Each element in the series should be list of strings

            stanza_pipeline:
                stanza.pipeline from prepare_stanza_pipeline

            min_ngram_length:
                setting for minimum number in ngram vectoriazation,
                used with custom analyzer
                default of 1

            max_ngram_length:
                setting for maximum number in ngram vectoriazation,
                used with custom analyzer
                default of 4

            sklearn_type:
                Setting for OneHotEncode, Regular CountVectorization, or TFIDFVectorization
                default for OneHotEncode, choose between "OneHotEncode", "CountVectorizer", "TFIDF"

        Returns:
            sklearn_transformer:
                sklearn text transformer for usage later/in MLflow models

            transformed_text:
                pd.DataFrame that combines the vectorized text with the original dataframe
        """

        sklearn_transformer_params = {
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "min_df": min_df,
            "analyzer": CustomSKLearnAnalyzer().stanza_analyzer(
                stanza_pipeline=stanza_pipeline,
                min_ngram_length=min_ngram_length,
                max_ngram_length=max_ngram_length,
            ),
            "sklearn_type": sklearn_type,
        }

        if sklearn_type == "OneHotEncode":
            sklearn_transformer_params["binary"] = True
            sklearn_transformer = CountVectorizer(**sklearn_transformer_params)

        elif sklearn_type == "CountVectorizer":
            print("/n")
            print(
                "Using CountVectorizer, but is not OneHotEncoded or TFIDF transformed"
            )
            sklearn_transformer_params["binary"] = False
            sklearn_transformer = CountVectorizer(**sklearn_transformer_params)

        elif sklearn_type == "TFIDF":
            sklearn_transformer_params["binary"] = False
            sklearn_transformer = TfidfVectorizer(**sklearn_transformer_params)

        else:
            print("/n")
            print(
                "Invalid sklearn text processing type, please choose between 'OneHotEncode', 'CountVectorizer', 'TFIDF'"
            )
            return None

        sklearn_transformer.fit(input_data)

        return sklearn_transformer

    @classmethod
    def stanza_analyzer(self, stanza_pipeline, minNgramLength, maxNgramLength):
        """
        Custom ngram analyzer function, matching only ngrams that belong to the same line

        The source for this was StackOverflow because I couldn't figure out how to let sklearn pipelines use arguments for custom analyzers

        Use this as the analyzer for an sklearn pipeline, and it should work

        Args:
            stanza_pipeline: Stanza pipeline
            minNgramLength: integer for the minimum ngram (usually 1)
            maxNgramLength: integer for maximum length ngram (usually should not exceed 4)

        Returns:
            A function that will be used in sklearn pipeline. Said function yields a generator

        """

        def ngrams_per_line(ingredients_list):

            lowered = " brk ".join(
                map(str, [ingred for ingred in ingredients_list if ingred is not None])
            ).lower()

            if lowered is None:
                lowered = "Missing ingredients"

            preproc = stanza_pipeline(lowered)

            lemmad = " ".join(
                map(
                    str,
                    [
                        word.lemma
                        for sent in preproc.sentences
                        for word in sent.words
                        if (
                            word.upos
                            not in ["NUM", "DET", "ADV", "CCONJ", "ADP", "SCONJ"]
                            and word is not None
                        )
                    ],
                )
            )

            # analyze each line of the input string seperately
            for ln in lemmad.split(" brk "):

                # tokenize the input string (customize the regex as desired)
                at_least_two_english_characters_whole_words = "(?u)\b[a-zA-Z]{2,}\b"
                terms = re.split(at_least_two_english_characters_whole_words, ln)

                # loop ngram creation for every number between min and max ngram length
                for ngramLength in range(minNgramLength, maxNgramLength + 1):

                    # find and return all ngrams
                    # for ngram in zip(*[terms[i:] for i in range(3)]):
                    # <-- solution without a generator (works the same but has higher memory usage)
                    for ngram in zip(
                        *[
                            islice(seq, i, len(terms))
                            for i, seq in enumerate(tee(terms, ngramLength))
                        ]
                    ):  # <-- solution using a generator

                        ngram = " ".join(map(str, ngram))
                        yield ngram

        return ngrams_per_line
