from pydantic.dataclasses import dataclass


# BACKEND
#
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
