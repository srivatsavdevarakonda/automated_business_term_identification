import os
import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from groq import Groq

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


# -------------------------------
# RULE BASED TERM MAPPING
# -------------------------------
def rule_based_term(column):

    col = column.lower()

    if "init_price" in col or "initial_price" in col:
        return "Initial Price"

    if "final_price" in col:
        return "Final Price"

    if "price" in col:
        return "Final Price"

    if "title" in col or "product_name" in col or "heading" in col:
        return "Product Name"

    if "brand" in col:
        return "Brand"

    if "rating" in col:
        return "Rating"

    if "currency" in col:
        return "Currency"

    if "url" in col:
        return "URL"

    if "review" in col:
        return "Reviews Count"

    if "category" in col:
        return "Category"

    if "seller_name" in col:
        return "Seller Name"

    if "seller_id" in col:
        return "Seller ID"

    return None


# -------------------------------
# NORMALIZE CONFIDENCE
# -------------------------------
def normalize_confidence(conf):

    try:
        c = float(conf)

        if c < 0:
            return 0.0

        if c > 1:
            return 1.0

        return round(c, 4)

    except:
        return 0.0


# -------------------------------
# CALL LLM
# -------------------------------
def call_llm(prompt: str) -> dict:
    """Call Groq LLM and return parsed JSON."""
    if not client.api_key:
        return {"term": "", "confidence": 0.0, "reason": "No API key"}

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )

        text = resp.choices[0].message.content.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])

    except Exception as e:
        print("⚠ LLM skipped due to rate limit")
        return {"term": "", "confidence": 0.0, "reason": "rate_limit"}

    return {"term": "", "confidence": 0.0, "reason": "parse_error"}

# -------------------------------
# BUILD PROMPT
# -------------------------------
def build_prompt(card_text, candidates):

    lines = []

    for i, c in enumerate(candidates, start=1):
        lines.append(f"{i}) {c['term']} - {c['definition']}")

    candidates_block = "\n".join(lines)

    return f"""
You are a data governance assistant.

Choose the BEST matching business term from the candidates.

COLUMN DETAILS:
{card_text}

CANDIDATE TERMS:
{candidates_block}

Return STRICT JSON with EXACT format:

{{
"term": "<one candidate term EXACTLY>",
"confidence": <a number between 0.0 and 1.0>,
"reason": "<10-20 word explanation>"
}}

Rules:
- Confidence MUST be a FLOAT between 0.0 and 1.0
- Initial Price = price before discount
- Final Price = price after discount
- Use column samples, hints and datatype
""".strip()


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():

    cards = pd.read_csv(RESULTS_DIR / "column_cards.csv")
    matches = pd.read_csv(RESULTS_DIR / "column_matches.csv")
    glossary = pd.read_csv(DATA_DIR / "glossary.csv")
    gold = pd.read_csv(RESULTS_DIR / "gold_labels.csv")

    # Only evaluate gold label columns
    gold_pairs = set(zip(gold["table"], gold["column"]))

    term_def = {
        row["TERM"]: row["DEFINITION"]
        for _, row in glossary.iterrows()
    }

    results = []

# Load previous predictions if file exists (to avoid re-calling LLM)
    pred_path = RESULTS_DIR / "llm_predictions.csv"

    existing_preds = {}

    if pred_path.exists():
        old_df = pd.read_csv(pred_path)
        for _, r in old_df.iterrows():
            existing_preds[(r["table"], r["column"])] = {
                "llm_term": r["llm_term"],
                "llm_confidence": r["llm_confidence"],
                "llm_reason": r["llm_reason"],
            }

    for (table, column), group in matches.groupby(["table", "column"]):
        # Skip if prediction already exists
        if (table, column) in existing_preds:
            prev = existing_preds[(table, column)]

            print(f"{table}.{column} → {prev['llm_term']} (cached)")

            results.append({
                "table": table,
                "column": column,
                "llm_term": prev["llm_term"],
                "llm_confidence": prev["llm_confidence"],
                "llm_reason": prev["llm_reason"]
            })

            continue

        # Skip non evaluation columns
        if (table, column) not in gold_pairs:
            continue

        # ---------------------------
        # RULE BASED FIRST
        # ---------------------------
        rule_term = rule_based_term(column)

        if rule_term:

            print(f"{table}.{column} → {rule_term} (rule)")

            results.append(
                {
                    "table": table,
                    "column": column,
                    "llm_term": rule_term,
                    "llm_confidence": 0.99,
                    "llm_reason": "Rule based mapping",
                }
            )

            continue

        # ---------------------------
        # LLM RERANKING
        # ---------------------------
        group = group.sort_values("rank").head(3)

        card_text = cards[
            (cards["table"] == table) & (cards["column"] == column)
        ].iloc[0]["card_text"]

        candidates = []

        for _, r in group.iterrows():

            candidates.append(
                {
                    "term": r["term"],
                    "definition": term_def.get(r["term"], ""),
                    "score": float(r["score"]),
                }
            )

        prompt = build_prompt(card_text, candidates)

        result = call_llm(prompt)

        raw_term = result.get("term", "")

        clean_term = raw_term.split("-")[0].strip()
        clean_term = clean_term.split("—")[0].strip()
        clean_term = clean_term.split(":")[0].strip()

        print(f"{table}.{column} → {clean_term}")

        results.append(
            {
                "table": table,
                "column": column,
                "llm_term": clean_term,
                "llm_confidence": normalize_confidence(
                    result.get("confidence", 0)
                ),
                "llm_reason": result.get("reason", ""),
            }
        )

    out_df = pd.DataFrame(results)

    out_df.to_csv(
        RESULTS_DIR / "llm_predictions.csv",
        index=False,
    )

    print("\nSaved to results/llm_predictions.csv")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    main()