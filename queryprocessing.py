import T5Tokenizer, T5ForConditionalGeneration
import spacy
import time
import warnings
from keybert import KeyBERT
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT('distilbert-base-nli-mean-tokens')

def filter_keywords(query):
    '''
    This function takes a query as input and extracts keywords using spaCy and KeyBERT.
    :param query: a string query longer than 3 characters.
    :return: list of keywords extracted from the query.
    '''
    assert isinstance(query, str), "Input query must be a string."
    assert len(query) > 0, "Input query must not be empty."
    if len(query) < 4:
        return query

    doc = nlp(query)

    spacy_keywords = [ent.text for ent in doc.ents]
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # Extract keywords with KeyBERT
    keybert_keywords = kw_model.extract_keywords(query, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)
    keybert_keywords = [kw[0] for kw in keybert_keywords]

    # Combine, deduplicate, and filter keywords
    all_keywords = set(spacy_keywords + noun_chunks + keybert_keywords)

    # Further filter keywords to focus on nouns and proper nouns
    filtered_keywords = [word for word in all_keywords if any(token.pos_ in ['NOUN', 'PROPN'] for token in nlp(word))]

    # Print the final keywords
    print("Final Keywords:", filtered_keywords)

    return filtered_keywords


if __name__ == "__main__":

    text = ["I would like to go somewhere maybe Tübingen. Lets say can you drive a boat in Tübingen? Maybe on the Neckar river.",
            "SUP tübingen neckar",
            "where is townhall",
            "where is the townhall",
            "where is the townhall located"]
    for t in text:
        filter_keywords(t)
        print("\n")

