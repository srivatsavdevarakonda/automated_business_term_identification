import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

def main():
    gold = pd.read_csv(RESULTS_DIR / "gold_labels.csv")
    llm = pd.read_csv(RESULTS_DIR / "llm_predictions.csv")

    merged = gold.merge(
        llm[["table", "column", "llm_term", "llm_confidence"]],
        on=["table", "column"],
        how="left",
    )

    # keep only those columns that have a correct_term
    eval_df = merged[
        merged["correct_term"].notna() & (merged["correct_term"] != "")
    ].copy()

    eval_df["is_correct"] = eval_df["correct_term"] == eval_df["llm_term"]

    total = len(eval_df)
    correct = int(eval_df["is_correct"].sum())
    accuracy = correct / total if total else 0.0

    print(f"Evaluated columns: {total}")
    print(f"Correct LLM predictions: {correct}")
    print(f"Top-1 LLM accuracy: {accuracy*100:.2f}%")
    print()
    print(
        eval_df[
            ["table", "column", "correct_term", "llm_term", "llm_confidence", "is_correct"]
        ]
    )

if __name__ == "__main__":
    main()
