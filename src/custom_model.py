"""
embedding_generation/app/custom_model.py

Updated CustomSKLearnPythonModel:
  - Stanza pipeline is NOT pickled — re-initialized in load_context(),
    with graceful download-on-miss (mirrors the embedding service lifespan)
  - predict() now lemmatizes input text via Stanza before calling
    sklearn_transformer.transform(), matching your training preprocessing
  - model_input expected as a polars/pandas Series of raw (non-lemmatized)
    ingredient strings — lemmatization happens inside predict(), not
    upstream, so callers don't need to know about it

Keep this as its own module (not inline in app.py) — mlflow.pyfunc.log_model
needs to import the class by reference via code_path, and a separate file
makes that import path stable across environments.

This is shared with the embedding_model/app/custom_model.py in the
embedding_service container
"""

import stanza
import mlflow.pyfunc
import polars as pl


class CustomSKLearnPythonModel(mlflow.pyfunc.PythonModel):
    """
    Wraps a trained sklearn TF-IDF transformer + the Stanza pipeline
    used to lemmatize ingredients before vectorizing them.

    Artifacts expected at log time:
      sklearn_transformer: pickled sklearn TfidfVectorizer (fitted)

    Stanza is intentionally NOT an artifact — it's downloaded/initialized
    fresh in load_context(). This keeps the MLflow model artifact small
    and avoids pickling a large, version-sensitive C-extension-backed object.
    """

    def load_context(self, context):
        import dill as pickle

        with open(context.artifacts["sklearn_transformer"], "rb") as f:
            self.sklearn_transformer = pickle.load(f)

        # Re-initialize Stanza rather than unpickling it. Downloads the
        # model on first load if not already cached in this environment
        # (matches the pattern used in embedding_generation's lifespan).
        try:
            self.nlp = stanza.Pipeline(
                "en",
                processors="tokenize,pos,lemma",
                verbose=False,
                download_method=None,  # use cached models if present
            )
        except Exception:
            stanza.download("en", processors="tokenize,pos,lemma")
            self.nlp = stanza.Pipeline(
                "en",
                processors="tokenize,pos,lemma",
                verbose=False,
            )

    def _lemmatize(self, text: str) -> str:
        """
        Lemmatize a single ingredient string, matching the preprocessing
        used during training. Joins lemmas with spaces.
        """
        doc = self.nlp(text)
        lemmas = [
            str(word.lemma)
            for sentence in doc.sentences
            for word in sentence.words
            if (
                word.upos not in ["NUM", "DET", "ADV", "CCONJ", "ADP", "SCONJ", "PUNCT"]
                and word is not None
            )
        ]
        return " ".join(lemmas)

    def predict(self, context, model_input):
        """
        model_input: pandas Series/DataFrame column of raw ingredient
        strings (one long string per recipe, NOT pre-lemmatized).

        Lemmatizes each entry, then transforms via the fitted TF-IDF
        vectorizer. Returns a DataFrame of TF-IDF scores with feature
        names as columns — same shape the original implementation produced.
        """
        raw_values = (
            model_input.values if hasattr(model_input, "values") else model_input
        )

        lemmatized = [self._lemmatize(str(text)) for text in raw_values]

        response = self.sklearn_transformer.transform(lemmatized)

        return pl.from_numpy(
            data=response.toarray(),
            schema=self.sklearn_transformer.get_feature_names_out().tolist(),
        )
