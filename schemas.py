from enum import Enum
import mlflow
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic.dataclasses import dataclass
import re
from typing import Dict, List, Optional

# from typing_extensions import TypedDict


# BACKEND


class CuisineChoices(Enum):
    AFRICAN = "African"
    AMERICAN = "American"
    ASIAN = "Asian"
    CAJCRE = "Cajun/Creole"
    CHINESE = "Chinese"
    EASTERNEURO = "Eastern European"
    ENGLISH = "English"
    FILIPINO = "Filipino"
    FRENCH = "French"
    GERMAN = "German"
    INDIAN = "Indian"
    IRISH = "Irish"
    ITALIAN = "Italian"
    JAPANESE = "Japanese"
    KOSHER = "Kosher"
    LATAM = "Latin American"
    MEDITERRANEAN = "Mediterranean"
    MEXICAN = "Mexican"
    MIDEAST = "Middle Eastern"
    MOROCCAN = "Moroccan"
    SCANDINAVIAN = "Scandinavian"
    SOUTHWESTERN = "Southwestern"
    THAI = "Thai"
    VIET = "Vietnamese"


class Edamam_API_Link(BaseModel):
    # if Edamam API starts implementing previous/next link in their search results, this could help
    href: str  # url for Edamam API response link
    title: str  # title of url/link


class Edamam_API_Link_Handler(BaseModel):
    next: Edamam_API_Link
    previous: Optional[Edamam_API_Link] = None


class Edamam_Recipe_Link(BaseModel):
    uri: str


class Edamam_Recipe_Image_Info(BaseModel):
    url: str
    width: int
    height: int


class Edamam_Recipe_Image_Collection(BaseModel):
    THUMBNAIL: Optional[Edamam_Recipe_Image_Info]
    SMALL: Optional[Edamam_Recipe_Image_Info]
    REGULAR: Optional[Edamam_Recipe_Image_Info]
    LARGE: Optional[Edamam_Recipe_Image_Info]


class Edamam_Ingredient(BaseModel):
    text: str
    quantity: float
    measure: str
    food: str
    weight: float
    foodCategory: str
    foodId: str
    image: str


class Edamam_Nutrient_Info(BaseModel):
    label: str
    quantity: float
    unit: str


class Edamam_Nutrients(BaseModel):
    ENERC_KCAL: Edamam_Nutrient_Info
    FAT: Edamam_Nutrient_Info
    FASAT: Edamam_Nutrient_Info
    FATRN: Edamam_Nutrient_Info
    FAMS: Edamam_Nutrient_Info
    FAPU: Edamam_Nutrient_Info
    CHOCDF: Edamam_Nutrient_Info
    CHOCDFnet_field: Edamam_Nutrient_Info = Field(..., alias="COCDF.net")
    FIBTG: Edamam_Nutrient_Info
    SUGAR: Edamam_Nutrient_Info
    PROCNT: Edamam_Nutrient_Info
    CHOLE: Edamam_Nutrient_Info
    NA: Edamam_Nutrient_Info
    CA: Edamam_Nutrient_Info
    MG: Edamam_Nutrient_Info
    K: Edamam_Nutrient_Info
    FE: Edamam_Nutrient_Info
    ZN: Edamam_Nutrient_Info
    P: Edamam_Nutrient_Info
    VITA_RAE: Edamam_Nutrient_Info
    VITC: Edamam_Nutrient_Info
    THIA: Edamam_Nutrient_Info
    RIBF: Edamam_Nutrient_Info
    NIA: Edamam_Nutrient_Info
    VITB6A: Edamam_Nutrient_Info
    FOLDFE: Edamam_Nutrient_Info
    FOLFD: Edamam_Nutrient_Info
    FOLAC: Edamam_Nutrient_Info
    VITB12: Edamam_Nutrient_Info
    VITD: Edamam_Nutrient_Info
    TOCPHA: Edamam_Nutrient_Info
    VITK1: Edamam_Nutrient_Info
    WATER: Edamam_Nutrient_Info

    model_config = ConfigDict(populate_by_name=True)


class Edamam_Nutrition_Metainfo(BaseModel):
    label: str
    tag: str
    schemaOrgTag: str
    total: float
    hasRDI: bool
    daily: float
    unit: str


class Edamam_Nutrition_Metainfo_Sub(Edamam_Nutrition_Metainfo):
    sub: Optional[List[Edamam_Nutrition_Metainfo]]


class Edamam_Recipe_Links_Handler(BaseModel):
    self: Edamam_API_Link


class Edamam_Recipe(BaseModel):
    recipe: Edamam_Recipe_Link
    label: str
    image: str
    images: Edamam_Recipe_Image_Collection
    source: str
    url: str
    shareAs: str
    yield_field: float = Field(..., alias="yield")
    dietLabels: List[str]
    healthLabels: List[str]
    cautions: List[str]
    ingredientLines: List[str]
    ingredients: List[Edamam_Ingredient]
    calories: float
    totalCO2Emissions: float
    totalWeight: float
    totalTime: float
    cuisineType: List[str]
    mealType: List[str]
    dishType: List[str]
    totalNutrients: Edamam_Nutrients
    totalDaily: Edamam_Nutrients
    digest: List[Edamam_Nutrition_Metainfo_Sub]
    _links: Edamam_Recipe_Links_Handler

    model_config = ConfigDict(populate_by_name=True)


class Edamam_API_Response(BaseModel):
    from_field: int  # starting index of returned results
    to_field: int  # ending index of returned results
    count_field: int  # total number of returned results

    # _links: Dict[str, Dict[str, Edamam_API_Link]] # any links from returned results, usually leading to a next page
    # testing the next link shows that there is no going backwards in responses. the link only goes to the next batch
    # given this, I'm not going to stress about the next/previous link and in case there aren't a lot of responses, this should maybe be optional anyway

    _links: Optional[Edamam_API_Link_Handler]
    hits: List[Edamam_Recipe]


# QUERY
class RecipeQuery(BaseModel):
    recipe_title: str


class UserRecipeQuery(RecipeQuery):
    cuisine_label: CuisineChoices  # moving forward, might be better to have this be List[CuisineChoices] to account for fusions/hybrids when scraping


class ScraperRecipeQuery(RecipeQuery):
    # use for scraping and inputting into database
    cuisine_labels: List[
        CuisineChoices
    ]  # as from EdamamAPI, can store multiple labels this way


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
