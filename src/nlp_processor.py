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
from typing import List, Text


class NLP_Processor:
    """
    This class is needed to add a custom tokenizer in sklearn that is pickled for use in other steps in an MLFlow Project. Import this class to gain access to the tokenizer and this can all be pickled.
    """

    def __init__(self, pretrained_str: Text) -> None:  # model_uri: str):
        """Initialize an instance with the MLFlow URI for a spaCy model
        Args:
            URI for MLFlow.spacy model"""

        self.nlp = spacy.load(pretrained_str)

    def custom_preprocessor(self, recipe_ingreds: Text) -> Any: #List[Token]:
        """This function replaces the default sklearn CountVectorizer preprocessor to use spaCy. sklearn CountVectorizer's preprocessor only performs accent removal and lowercasing.

        Args:
            A string to tokenize from a recipe representing the ingredients used in the recipe

        Returns:
            A list of strings that have been de-accented and lowercased to be used in tokenization
        """
        preprocessed = [token for token in self.nlp(recipe_ingreds.lower())]

        return preprocessed

    def custom_lemmatizer(ingredients:List) -> Any: 
        """This takes in a string representing the recipe and an NLP model and lemmatize with the NER. 
    
    Pronouns (like "I" and "you" get lemmatized to '-PRON-', so I'm removing those.
    Remove punctuation

    Args:
        ingredients: string
        nlp_mod: spacy model (try built in first, by default called nlp)
    
    Returns:
        List[String]
    """
    lemmas = [token.lemma_ for token in ingredients if (token.is_alpha and token.pos_ not in ["PRON", "VERB"] and len(token.lemma_) > 1)]
    return lemmas
