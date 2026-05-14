import stanza
from sentence_transformers import SentenceTransformer

print("Downloading stanza English model...")
stanza.download("en", processors="tokenize,pos,lemma,ner")

print("Downloading sentence_transformers model...")
# all-MiniLM-L6-v2: 384 dimensions, fast, good quality — matches your schema
SentenceTransformer("all-MiniLM-L6-v2")

print("All models downloaded.")
