from itertools import tee, islice
import mlflow
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import stanza
import src.custom_sklearn_text_transformer_mlflow


class CustomSKLearnWrapper(mlflow.pyfunc.PythonModel):
    """
    This class allows Stanza pipelines to be logged in MLflow as a
    custom PythonModel
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

        with open(context.artifacts["sklearn_model"], "rb") as f:
            self.model = pickle.load(f)

        with open(context.artifacts["sklearn_transformer"], "rb") as f:
            self.sklearn_transformer = pickle.load(f)

    def predict(self, context, model_input, params):
        """
        This method is needed to override the default predict.
        It needs to function essentially as a wrapper and returns back the
        Transformer or Transformer Pipeline itself

        Args:
            context:        Any
                Not used

            model:          sklearn.Transformer
                The sklearn text Transformer or Pipeline that ends in a
                Transformer

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

        # response = self.sklearn_transformer.transform(model_input)
        response = self.sklearn_transformer.transform(model_input.values)

        transformed_recipe = pd.DataFrame(
            response.toarray(),
            columns=self.sklearn_transformer.get_feature_names_out(),
            index=model_input.index,
        )

        return transformed_recipe
