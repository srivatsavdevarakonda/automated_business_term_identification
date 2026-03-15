import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

def normalize(x):
    return str(x).strip().lower().replace("_", " ")

def build_synonym_map(glossary):
    syn_map = {}

    for _, row in glossary.iterrows():
        term = normalize(row["TERM"])
        syns = str(row.get("SYNONYMS", "")).lower().split(",")

        syn_list = [normalize(s) for s in syns if s.strip() != ""]
        syn_list.append(term)

        syn_map[term] = syn_list

    return syn_map


def is_match(correct, predicted, syn_map):

    correct = normalize(correct)
    predicted = normalize(predicted)

    if correct == predicted:
        return True

    if correct in syn_map and predicted in syn_map[correct]:
        return True

    return False


def main():

    gold = pd.read_csv(RESULTS_DIR / "gold_labels.csv")
    llm = pd.read_csv(RESULTS_DIR / "llm_predictions.csv")
    matches = pd.read_csv(RESULTS_DIR / "column_matches.csv")
    glossary = pd.read_csv(DATA_DIR / "glossary.csv")

    syn_map = build_synonym_map(glossary)

    # Merge LLM predictions
    merged = gold.merge(
        llm[["table", "column", "llm_term", "llm_confidence"]],
        on=["table", "column"],
        how="left",
    )

    # Only evaluate rows with correct label
    eval_df = merged[
        merged["correct_term"].notna() & (merged["correct_term"] != "")
    ].copy()

    # Top-1 correctness using synonyms
    eval_df["top1_correct"] = eval_df.apply(
        lambda r: is_match(r["correct_term"], r["llm_term"], syn_map),
        axis=1
    )

    # Get Top-3 candidate terms
    top3 = matches[matches["rank"] <= 3]

    top3_terms = (
        top3.groupby(["table", "column"])["term"]
        .apply(list)
        .reset_index()
        .rename(columns={"term": "top3_terms"})
    )

    eval_df = eval_df.merge(top3_terms, on=["table", "column"], how="left")

    # Top-3 correctness with synonym check
    def check_top3(row):
        if not isinstance(row["top3_terms"], list):
            return False

        for t in row["top3_terms"]:
            if is_match(row["correct_term"], t, syn_map):
                return True

        return False

    eval_df["top3_correct"] = eval_df.apply(check_top3, axis=1)

    total = len(eval_df)

    top1_correct = int(eval_df["top1_correct"].sum())
    top3_correct = int(eval_df["top3_correct"].sum())

    top1_accuracy = top1_correct / total if total else 0
    top3_accuracy = top3_correct / total if total else 0

    # Precision / Recall / F1
    precision = top1_accuracy
    recall = top1_accuracy
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    print(f"Evaluated columns: {total}")
    print(f"Correct LLM predictions (Top-1): {top1_correct}")
    print(f"Top-1 Accuracy: {top1_accuracy*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy*100:.2f}%")
    print()
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall: {recall*100:.2f}%")
    print(f"F1 Score: {f1*100:.2f}%")
    print()

    print(
        eval_df[
            ["table","column","correct_term","llm_term","llm_confidence","top1_correct","top3_correct"]
        ]
    )


if __name__ == "__main__":
    main()