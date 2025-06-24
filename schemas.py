import mlflow
import pandas as pd
from pydantic.dataclasses import dataclass
import re
from typing import List

# BACKEND


@dataclass
class CustomSKLearnAnalyzer(dataclass):
    """
    This class handles allows sklearn text transformers to incorporate a Stanza pipeline with a custom analyzer
    """

    # Options: require a stanza_lang_str which will be used to download a Stanza model
    stanza_lang_str: str = "en"
    depparse_batch_size: int = 50
    depparse_min_length_to_batch_separately: int = 50
    verbose: bool = True
    use_gpu: bool = True
    batch_size: int = 100

    @classmethod
    def ngram_maker(self, min_ngram_length: int, max_ngram_length: int):
        def ngrams_per_line(row: str):
            for ln in row.split(" brk "):
                at_least_two_english_characters_whole_words = r"(?u)\b\w{2,}\b"
                terms = re.findall(at_least_two_english_characters_whole_words, ln)
                for ngram_length in range(min_ngram_length, max_ngram_length + 1):

                    # find and return all ngrams
                    # for ngram in zip(*[terms[i:] for i in range(3)]):
                    # <-- solution without a generator (works the same but has higher memory usage)
                    for ngram in (
                        word
                        for i in range(len(terms) - ngram_length + 1)
                        for word in (" ".join(terms[i : i + ngram_length]),)
                    ):
                        yield ngram

        return ngrams_per_line

    # TODO
    # Is it possible to move the download of the model into the container creation, and require the Stanza model in this instantiation instead
    # stanza_model: StanzaModel
    # stanza documentation here (https://github.com/stanfordnlp/stanza-train) implies that the pretrained models are PyTorch models
    # having some difficulty finding examples of creating a BaseModel from a PyTorch model. Switch to string option above


# SKLEARN processing


@dataclass
class CustomSKLearnWrapper(mlflow.pyfunc.PythonModel):
    """
    This class allows sklearn text transformers to be logged in MLflow as a
    custom PythonModel. It overrides the default load_context and predict methods (as required by MLflow).
    load_context now loads pickled files representing the model itself (which requires Stanza) and the transformer (which is an sklearn object)
    """

    # def __init__(self, model):
    #     """
    #     Constructor method. Initializes the model with a Stanza libary language
    #     type. The default is "en" for English

    #     model:          sklearn.Transformer
    #             The sklearn text Transformer or Pipeline that ends in a
    #             Transformer

    #     later can add functionality to include pretrained models needed for Stanza

    #     """
    #     self.model = model

    def load_context(self, context):
        """
        Method needed to override default load_context. Needs to handle different components of sklearn model

        """
        import dill as pickle

        # dill is needed due to generators and classes in the model itself

        with open(context.artifacts["sklearn_model"], "rb") as f:
            self.model = pickle.load(f)

        with open(context.artifacts["sklearn_transformer"], "rb") as f:
            self.sklearn_transformer = pickle.load(f)

    def predict(self, context, model_input: List[str], params: dict):
        """
        This method is needed to override the default predict.
        It needs to function essentially as a wrapper and returns back the
        transformed recipes

        Args:
            context:        Any
                Not used

            model_input:    List(string)
                The ingredients of a single query recipe in a list
                Need to decide if this is taking in raw text or preprocessed text
                Leaning towards taking in raw text, doing preprocessing, and
                logging the pre processed text as an artifact

            params:         dict, optional
                Parameters used for the model (optional)
                Not used currently for sklearn

        Returns:
            transformed_recipe_df: DataFrame of the recipes after going through
            the sklearn/Stanza text processing
        """

        print(model_input)
        print(model_input.shape)
        print(model_input.sample(3, random_state=200))

        response = self.sklearn_transformer.transform(model_input.values)

        transformed_recipe = pd.DataFrame(
            response.toarray(),
            columns=self.sklearn_transformer.get_feature_names_out(),
            index=model_input.index,
        )

        return transformed_recipe
