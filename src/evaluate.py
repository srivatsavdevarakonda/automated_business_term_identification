import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")


def main():
    # load top-k matches from retrieve step
    matches_df = pd.read_csv(RESULTS_DIR / "column_matches.csv")

    # keep only rank-1 as model prediction
    top1 = matches_df[matches_df["rank"] == 1].copy()
    top1.rename(columns={"term": "predicted_term"}, inplace=True)

    # load gold labels (true terms)
    gold_df = pd.read_csv(RESULTS_DIR / "gold_labels.csv")

    # join on table + column
    merged = pd.merge(
        gold_df,
        top1,
        on=["table", "column"],
        how="left",
    )

    # ignore rows where there is no correct_term (empty)
    eval_df = merged[merged["correct_term"].notna() & (merged["correct_term"] != "")].copy()

    # compute correctness
    eval_df["is_correct"] = eval_df["correct_term"] == eval_df["predicted_term"]

    total = len(eval_df)
    correct = int(eval_df["is_correct"].sum())
    accuracy = correct / total if total > 0 else 0.0

    print(f"Evaluated columns: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Top-1 accuracy: {accuracy*100:.2f}%")
    print()
    print("Detailed results:")
    print(
        eval_df[
            ["table", "column", "correct_term", "predicted_term", "score", "is_correct"]
        ]
    )


if __name__ == "__main__":
    main()
