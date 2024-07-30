from itertools import tee, islice
import re
import stanza


class CustomSKLearnAnalyzer:
    """
    This class handles allows sklearn text transformers to incorporate a Stanza pipeline with a custom analyzer
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
        use_gpu=True,
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
