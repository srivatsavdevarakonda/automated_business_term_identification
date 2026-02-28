import pandas as pd
import re
import json
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def profile_df(df: pd.DataFrame, table_name: str):
    """Return a list of column profile dicts for a given table."""
    cards = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        dtype = str(series.dtype)
        row_count = len(series)
        null_pct = round(float(series.isna().mean()), 3) if row_count else 1.0
        distinct = int(non_null.nunique()) if row_count else 0

        samples = [str(v) for v in non_null.unique()[:5]]

        hints = []
        sample_text = " ".join(samples)

        if re.search(r"@", sample_text):
            hints.append("email_like")
        if re.search(r"\d{4}-\d{2}-\d{2}", sample_text):
            hints.append("date_like")
        if re.search(r"^\+?\d{7,15}$", sample_text.replace(" ", "")):
            hints.append("phone_like")
        if dtype.startswith(("int", "float")):
            hints.append("numeric_dtype")

        cards.append(
            {
                "table": table_name,
                "column": col,
                "dtype": dtype,
                "row_count": row_count,
                "null_pct": null_pct,
                "distinct": distinct,
                "samples": samples,
                "hints": hints,
            }
        )

    return cards


def card_to_text(card: dict) -> str:
    """Convert a column profile dict into a Column Card text string."""
    samples_str = ", ".join(card["samples"])
    hints_str = ", ".join(card["hints"]) if card["hints"] else "none"

    return (
        f"[Table] {card['table']}\n"
        f"[Column] {card['column']} ({card['dtype']})\n"
        f"[Stats] rows={card['row_count']}, null_pct={card['null_pct']}, distinct={card['distinct']}\n"
        f"[Samples] {samples_str}\n"
        f"[Hints] {hints_str}"
    )


def main():
    all_cards = []

    # Loop through ALL CSV files in data folder
    for csv_file in DATA_DIR.glob("*.csv"):

        # Skip glossary file
        if csv_file.name.lower() == "glossary.csv":
            continue

        print(f"Profiling table: {csv_file.name}")

        df = pd.read_csv(csv_file)
        table_name = csv_file.stem  # filename without .csv

        all_cards += profile_df(df, table_name)

    # Convert to text
    for card in all_cards:
        card["card_text"] = card_to_text(card)

    df_cards = pd.DataFrame(all_cards)
    df_cards.to_csv(RESULTS_DIR / "column_cards.csv", index=False)

    with open(RESULTS_DIR / "column_cards.json", "w", encoding="utf-8") as f:
        json.dump(all_cards, f, indent=2, ensure_ascii=False)

    print("\nGenerated column cards:")
    print(df_cards[["table", "column"]])


if __name__ == "__main__":
    main()