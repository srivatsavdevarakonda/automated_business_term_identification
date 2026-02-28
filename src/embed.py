import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def card_to_embedding_text(card_row):
    # normalize column name
    col_raw = str(card_row["column"])
    col_norm = col_raw.lower().replace("_", " ")

    # repeat column name to give it more weight
    return (
        f"Column: {col_norm} {col_norm} {col_norm}\n"
        f"Table: {card_row['table']}\n"
        f"Type: {card_row['dtype']}\n"
        f"Samples: {card_row['samples']}\n"
        f"Hints: {card_row['hints']}"
    )



def glossary_to_text(row):
    term = str(row.get("TERM", "")).strip()
    definition = str(row.get("DEFINITION", "")).strip()
    synonyms = str(row.get("SYNONYMS", "")).strip()

    term_norm = term.lower()

    return (
        f"Term: {term_norm} {term_norm}\n"
        f"Definition: {definition}\n"
        f"Synonyms: {synonyms}"
    )



def main():
    print("Loading column cards...")
    cards_df = pd.read_csv(RESULTS_DIR / "column_cards.csv")

    print("Loading glossary terms...")
    glossary_df = pd.read_csv(
    DATA_DIR / "glossary.csv",
    encoding="utf-8",
    sep=",",
    quotechar='"',
    escapechar="\\",
    engine="python",
    dtype=str
).fillna("")

    # Build texts
    card_texts = cards_df.apply(card_to_embedding_text, axis=1).tolist()
    glossary_texts = glossary_df.apply(glossary_to_text, axis=1).tolist()

    # Combine to train TF-IDF vocabulary
    all_texts = card_texts + glossary_texts

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    lowercase=True,
    )

    all_vectors = vectorizer.fit_transform(all_texts)

    # Split back into card vs glossary embeddings
    card_embeddings = all_vectors[: len(card_texts)].toarray()
    glossary_embeddings = all_vectors[len(card_texts) :].toarray()

    # Save embeddings
    np.save(RESULTS_DIR / "card_embeddings.npy", card_embeddings)
    np.save(RESULTS_DIR / "glossary_embeddings.npy", glossary_embeddings)

    print("Embeddings saved (TF-IDF)!")
    print("Card Embeddings Shape:", card_embeddings.shape)
    print("Glossary Embeddings Shape:", glossary_embeddings.shape)


if __name__ == "__main__":
    main()
