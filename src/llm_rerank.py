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

def normalize_confidence(conf):
    try:
        c = float(conf)
        if c < 0: return 0.0
        if c > 1: return 1.0
        return round(c, 4)
    except:
        return 0.0

def call_llm(prompt: str) -> dict:
    """Call Groq LLM and return parsed JSON."""
    if not client.api_key:
        raise RuntimeError("GROQ_API_KEY env var not set")

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
            return json.loads(text[start : end + 1])

    return {"term": "", "confidence": 0.0, "reason": text}


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
    - Confidence MUST be a FLOAT between 0.0 and 1.0.
    - Higher confidence means stronger semantic match.
    - Use column samples, hints, and datatype to justify the score.
    - Do NOT output definition in the term.
    """.strip()



def main():
    cards = pd.read_csv(RESULTS_DIR / "column_cards.csv")
    matches = pd.read_csv(RESULTS_DIR / "column_matches.csv")
    glossary = pd.read_csv(DATA_DIR / "glossary.csv")

    term_def = {row["TERM"]: row["DEFINITION"] for _, row in glossary.iterrows()}

    results = []

    for (table, column), group in matches.groupby(["table", "column"]):
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
        clean_term = raw_term.split("-")[0].strip()     # cut anything after "-"
        clean_term = clean_term.split("—")[0].strip()   # handles long dash
        clean_term = clean_term.split(":")[0].strip()   # handles colon
        result["term"] = clean_term

        print(f"{table}.{column} → {result.get('term')}")

        term = result.get("term", "")
        confidence = normalize_confidence(result.get("confidence", 0))
        reason = result.get("reason", "")


        results.append(
            {
                "table": table,
                "column": column,
                "llm_term": term,
                "llm_confidence": confidence,
                "llm_reason": reason,
            }
        )


    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULTS_DIR / "llm_predictions.csv", index=False)

    print("\nSaved to results/llm_predictions.csv")


if __name__ == "__main__":
    main()
