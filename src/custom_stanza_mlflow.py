from itertools import tee, islice
import mlflow
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import stanza


class StanzaWrapper(mlflow.pyfunc.PythonModel):
    """
    This class allows Stanza pipelines to be logged in MLflow as a
    custom PythonModel
    """

    def load_context(self, context):
        """
        This method is called when loading an MLflow model with
        pyfunc.load_model(), as soon as the Python Model is constructed.

        Args:
            context: MLflow context where the model artifact is stored.
        """
        import pickle

        self.model = pickle.load(open(context.artifacts["sklearn_transformer"], "rb"))
        self.database = pickle.load(open(context.artifacts["data"], "rb"))

    def predict(self, ingredients_list: list):
        """
        This method is needed to override the default predict.
        It needs to function essentially as a wrapper

        Args:
            the ingredients of a single, query recipe in a list

        Returns:
            similar_recipes_df: DataFrame of the top 5 most similar recipes from
            the database
        """

        response = self.model.transform(ingredients_list)

        transformed_recipe = pd.DataFrame(
            response.toarray(), columns=self.model.get_feature_names()
        )

        similar_recipes_df = self.find_closest_recipes(
            filtered_ingred_word_matrix=query_matrix,
            recipe_tfidf=self.model,
            X_df=prepped,
        )
        return similar_recipes_df
    
    # custom ngram analyzer function, matching only ngrams that belong to the same line
    def stanza_analyzer(stanza_pipeline, minNgramLength, maxNgramLength):
        def ngrams_per_line(ingredients_list):

            lowered = " brk ".join(map(str, [ingred for ingred in ingredients_list if ingred is not None])).lower()
            
            if lowered is None:
                lowered = "Missing ingredients"
            
            preproc = stanza_pipeline(lowered)
            
            lemmad = " ".join(map(str,
                                [word.lemma 
                                for sent in preproc.sentences 
                                for word in sent.words if (
                                    word.upos not in ["NUM", "DET", "ADV", "CCONJ", "ADP", "SCONJ"]
                                    #    and word not in STOP_WORDS
                                    and word is not None
                                )]
                            )
                        )
            
            # analyze each line of the input string seperately
            for ln in lemmad.split(' brk '):
                
                # tokenize the input string (customize the regex as desired)
                terms = re.split("(?u)\b[a-zA-Z]{2,}\b", ln)

                # loop ngram creation for every number between min and max ngram length
                for ngramLength in range(minNgramLength, maxNgramLength+1):

                    # find and return all ngrams
                    # for ngram in zip(*[terms[i:] for i in range(3)]): <-- solution without a generator (works the same but has higher memory usage)
                    for ngram in zip(*[islice(seq, i, len(terms)) for i, seq in enumerate(tee(terms, ngramLength))]): # <-- solution using a generator
                        
                        ngram = ' '.join(map(str, ngram))
                        yield ngram
                        
        return ngrams_per_line

    # def fit_transform(self, nlp_sklearn_params: list):
    #         """ 
    #         This method duplicates/wraps scikit-learn behavior for Pipelines to handle text

    #         Args:
    #             nlp_sklearn_params: list of tuples

    #         Returns:
    #             pipe: Pipeline
    #         """
    #     return t

    def filter_out_cuisine(ingred_word_matrix, X_df, cuisine_name, tfidf):
        # This function takes in the ingredient word matrix (from joblib), a
        # dataframe made from the database (from joblib), the user inputted cuisine
        # name, and the ingredient TFIDF Vectorizer object (from joblib) and returns
        # a word sub matrix that removes all recipes with the same cuisine as the
        # inputted recipe.

        combo = pd.concat([ingred_word_matrix, X_df["imputed_label"]], axis=1)
        filtered_ingred_word_matrix = combo[
            combo["imputed_label"] != cuisine_name
        ].drop("imputed_label", axis=1)
        return filtered_ingred_word_matrix

    def find_closest_recipes(filtered_ingred_word_matrix, recipe_tfidf, X_df):
        # This function takes in the filtered ingredient word matrix from function
        # filter_out_cuisine, the TFIDF recipe from function transform_tfidf, and
        # a dataframe made from the database (from joblib) and returns a Pandas
        # DataFrame with the top five most similar recipes and a Pandas Series
        # containing the similarity amount
        search_vec = np.array(recipe_tfidf).reshape(1, -1)
        res_cos_sim = cosine_similarity(filtered_ingred_word_matrix, search_vec)
        top_five = np.argsort(res_cos_sim.flatten())[-5:][::-1]
        proximity = res_cos_sim[top_five]
        recipe_ids = [filtered_ingred_word_matrix.iloc[idx].name for idx in top_five]
        suggest_df = X_df.loc[recipe_ids]
        suggest_df = pd.concat([suggest_df, proximity])
        return suggest_df
