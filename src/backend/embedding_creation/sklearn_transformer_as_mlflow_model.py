import mlflow
import pandas as pd
from typing import List


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
