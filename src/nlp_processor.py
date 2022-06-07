"""
This script creates a class with a custom scikit-learn tokenizer using spaCy. 

The class is necessary to preserve pickling via pickle or joblib a scikit-learn transformer.

Typical usage example:
Create custom class to be pickled and included with Scikit-learn CountVectorizer objects.

MVP: Just deliver the custom class to use in a pipeline.

Feature request: log in MLOps metrics (could be MLFlow could be MLOps side of DVC)
"""

# import mlflow.spacy
import spacy
from typing import List, String


class NLP_Processor:
    """
    This class is needed to add a custom tokenizer in sklearn that is pickled for use in other steps in an MLFlow Project. Import this class to gain access to the tokenizer and this can all be pickled.
    """

    def __init__(self, pretrained_str: String) -> None:  # model_uri: str):
        """Initialize an instance with the MLFlow URI for a spaCy model
        Args:
            URI for MLFlow.spacy model"""
        
        self.nlp = spacy.load(pretrained_str)

    def custom_preprocessor(self, recipe_ingreds: String) -> List:
        """This function replaces the default sklearn CountVectorizer preprocessor to use spaCy. sklearn CountVectorizer's preprocessor only performs accent removal and lowercasing.

        Args:
            A string to tokenize from a recipe representing the ingredients used in the recipe

        Returns:
            A list of strings that have been de-accented and lowercased to be used in tokenization
        """
        preprocessed = [
            token
            for token in self.nlp(recipe_ingreds.lower())
        ]

        return preprocessed

    def custom_tokenizer(self, report_text: String) -> List:
        """This function replaces the default sklearn CountVectorizer tokenizer to use spaCy. It also uses spaCy attributes for filtering. It expects a string and returns a list of tokenized strings.
        Args:
            A string to tokenize from a citation
        Returns:
            A list of tokens representing those strings"""

        lemmas = [
            token.lemma_
            for token in self.nlp(report_text.lower())
            if (token.is_alpha and token.lemma_ != "-PRON-" and len(token.lemma_) > 1)
        ]
        return lemmas