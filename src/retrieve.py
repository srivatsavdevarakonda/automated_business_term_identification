import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def cosine_similarity(a, b):
    """
    a: (n, d)
    b: (m, d)
    returns: (n, m) cosine similarity matrix
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def main():
    # Load embeddings
    card_emb = np.load(RESULTS_DIR / "card_embeddings.npy")
    gloss_emb = np.load(RESULTS_DIR / "glossary_embeddings.npy")

    # Load metadata
    cards_df = pd.read_csv(RESULTS_DIR / "column_cards.csv")
    glossary_df = pd.read_csv(DATA_DIR / "glossary.csv")

    # Compute cosine similarity: (num_cols, num_terms)
    sim = cosine_similarity(card_emb, gloss_emb)

    top_k = 3
    rows = []

    for i, card_row in cards_df.iterrows():
        # indices of top-k glossary terms for this column
        scores = sim[i]
        top_idx = scores.argsort()[::-1][:top_k]  # descending

        for rank, j in enumerate(top_idx, start=1):
            term = glossary_df.iloc[j]["TERM"]
            score = scores[j]
            rows.append(
                {
                    "table": card_row["table"],
                    "column": card_row["column"],
                    "rank": rank,
                    "term": term,
                    "score": float(score),
                }
            )

    matches_df = pd.DataFrame(rows)
    matches_path = RESULTS_DIR / "column_matches.csv"
    matches_df.to_csv(matches_path, index=False)

    print("Top-k matches saved to:", matches_path)
    print()
    print(matches_df)


if __name__ == "__main__":
    main()
