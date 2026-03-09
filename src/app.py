import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
import tempfile
import shutil
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Term Mapper",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "phase" not in st.session_state:
    st.session_state.phase = "upload"          # upload | pipeline | review
if "session_dir" not in st.session_state:
    st.session_state.session_dir = None
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

# ─────────────────────────────────────────────
# HELPERS – pipeline functions (adapted from src/)
# ─────────────────────────────────────────────

def profile_df(df: pd.DataFrame, table_name: str):
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
        cards.append({
            "table": table_name, "column": col, "dtype": dtype,
            "row_count": row_count, "null_pct": null_pct,
            "distinct": distinct, "samples": samples, "hints": hints,
        })
    return cards


def card_to_text(card: dict) -> str:
    samples_str = ", ".join(card["samples"])
    hints_str = ", ".join(card["hints"]) if card["hints"] else "none"
    return (
        f"[Table] {card['table']}\n"
        f"[Column] {card['column']} ({card['dtype']})\n"
        f"[Stats] rows={card['row_count']}, null_pct={card['null_pct']}, distinct={card['distinct']}\n"
        f"[Samples] {samples_str}\n"
        f"[Hints] {hints_str}"
    )


def run_profile(data_dir: Path, results_dir: Path, glossary_name: str):
    all_cards = []
    for csv_file in data_dir.glob("*.csv"):
        if csv_file.name.lower() == glossary_name.lower():
            continue
        df = pd.read_csv(csv_file)
        table_name = csv_file.stem
        all_cards += profile_df(df, table_name)
    for card in all_cards:
        card["card_text"] = card_to_text(card)
    df_cards = pd.DataFrame(all_cards)
    df_cards.to_csv(results_dir / "column_cards.csv", index=False)
    with open(results_dir / "column_cards.json", "w", encoding="utf-8") as f:
        json.dump(all_cards, f, indent=2, ensure_ascii=False)
    return len(all_cards)


def card_to_embedding_text(card_row):
    col_norm = str(card_row["column"]).lower().replace("_", " ")
    return (
        f"Column: {col_norm} {col_norm} {col_norm}\n"
        f"Table: {card_row['table']}\n"
        f"Type: {card_row['dtype']}\n"
        f"Samples: {card_row['samples']}\n"
        f"Hints: {card_row['hints']}"
    )


def glossary_to_text(row):
    term = str(row.get("TERM", "")).strip().lower()
    definition = str(row.get("DEFINITION", "")).strip()
    synonyms = str(row.get("SYNONYMS", "")).strip()
    return f"Term: {term} {term}\nDefinition: {definition}\nSynonyms: {synonyms}"


def run_embed(data_dir: Path, results_dir: Path, glossary_name: str):
    cards_df = pd.read_csv(results_dir / "column_cards.csv")
    glossary_df = pd.read_csv(
        data_dir / glossary_name,
        encoding="utf-8", sep=",", quotechar='"',
        escapechar="\\", engine="python", dtype=str
    ).fillna("")
    card_texts = cards_df.apply(card_to_embedding_text, axis=1).tolist()
    glossary_texts = glossary_df.apply(glossary_to_text, axis=1).tolist()
    all_texts = card_texts + glossary_texts
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    all_vectors = vectorizer.fit_transform(all_texts)
    card_embeddings = all_vectors[:len(card_texts)].toarray()
    glossary_embeddings = all_vectors[len(card_texts):].toarray()
    np.save(results_dir / "card_embeddings.npy", card_embeddings)
    np.save(results_dir / "glossary_embeddings.npy", glossary_embeddings)
    return card_embeddings.shape, glossary_embeddings.shape


def cosine_similarity(a, b):
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def run_retrieve(data_dir: Path, results_dir: Path, glossary_name: str):
    card_emb = np.load(results_dir / "card_embeddings.npy")
    gloss_emb = np.load(results_dir / "glossary_embeddings.npy")
    cards_df = pd.read_csv(results_dir / "column_cards.csv")
    glossary_df = pd.read_csv(data_dir / glossary_name, dtype=str).fillna("")
    sim = cosine_similarity(card_emb, gloss_emb)
    top_k = 3
    rows = []
    for i, card_row in cards_df.iterrows():
        scores = sim[i]
        top_idx = scores.argsort()[::-1][:top_k]
        for rank, j in enumerate(top_idx, start=1):
            rows.append({
                "table": card_row["table"], "column": card_row["column"],
                "rank": rank, "term": glossary_df.iloc[j]["TERM"],
                "score": float(scores[j]),
            })
    matches_df = pd.DataFrame(rows)
    matches_df.to_csv(results_dir / "column_matches.csv", index=False)
    return len(matches_df)


def run_llm_rerank(data_dir: Path, results_dir: Path, glossary_name: str, groq_api_key: str):
    from groq import Groq
    client = Groq(api_key=groq_api_key)

    def normalize_confidence(conf):
        try:
            c = float(conf)
            return max(0.0, min(1.0, round(c, 4)))
        except:
            return 0.0

    def call_llm(prompt):
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        try:
            return json.loads(text)
        except:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
        return {"term": "", "confidence": 0.0, "reason": text}

    cards = pd.read_csv(results_dir / "column_cards.csv")
    matches = pd.read_csv(results_dir / "column_matches.csv")
    glossary = pd.read_csv(data_dir / glossary_name, dtype=str).fillna("")
    term_def = {row["TERM"]: row["DEFINITION"] for _, row in glossary.iterrows()}
    results = []

    for (table, column), group in matches.groupby(["table", "column"]):
        group = group.sort_values("rank").head(3)
        card_text = cards[(cards["table"] == table) & (cards["column"] == column)].iloc[0]["card_text"]
        candidates = [{"term": r["term"], "definition": term_def.get(r["term"], ""), "score": float(r["score"])} for _, r in group.iterrows()]
        lines = [f"{i}) {c['term']} - {c['definition']}" for i, c in enumerate(candidates, 1)]
        prompt = f"""You are a data governance assistant.
Choose the BEST matching business term from the candidates.

COLUMN DETAILS:
{card_text}

CANDIDATE TERMS:
{chr(10).join(lines)}

Return STRICT JSON with EXACT format:
{{"term": "<one candidate term EXACTLY>", "confidence": <0.0-1.0>, "reason": "<10-20 word explanation>"}}

Rules:
- Confidence MUST be a FLOAT between 0.0 and 1.0.
- Use column samples, hints, and datatype to justify the score.
- Do NOT output definition in the term."""

        result = call_llm(prompt)
        raw_term = result.get("term", "")
        for sep in ["-", "—", ":"]:
            raw_term = raw_term.split(sep)[0].strip()
        results.append({
            "table": table, "column": column,
            "llm_term": raw_term,
            "llm_confidence": normalize_confidence(result.get("confidence", 0)),
            "llm_reason": result.get("reason", ""),
        })

    pd.DataFrame(results).to_csv(results_dir / "llm_predictions.csv", index=False)
    return len(results)


# ─────────────────────────────────────────────
# PHASE 1 – UPLOAD
# ─────────────────────────────────────────────
def show_upload_phase():
    st.markdown("""
    <div style="text-align:center; padding: 40px 0 10px 0;">
        <h1 style="font-size:2.5rem;">🧠 GenAI Term Mapper</h1>
        <p style="color:#9CA3AF; font-size:1.1rem;">
            Upload your e-commerce CSV datasets and your business glossary.<br>
            The pipeline will automatically map every column to a standardized business term.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📂 Upload Dataset CSVs")
        st.caption("Select one or more product CSV files (Amazon, Walmart, Shein, etc.)")
        dataset_files = st.file_uploader(
            "Drop your CSV files here",
            type=["csv"],
            accept_multiple_files=True,
            key="dataset_uploader",
            label_visibility="collapsed"
        )

    with col2:
        st.subheader("📖 Upload Glossary CSV")
        st.caption("Your business glossary with TERM, DEFINITION, SYNONYMS columns")
        glossary_file = st.file_uploader(
            "Drop your glossary CSV here",
            type=["csv"],
            accept_multiple_files=False,
            key="glossary_uploader",
            label_visibility="collapsed"
        )

    st.divider()

    # Groq API key
    st.subheader("🔑 Groq API Key")
    st.caption("Required for the LLM reranking step. Get a free key at console.groq.com")

    groq_key_env = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
    if groq_key_env:
        st.success("✅ Groq API key loaded from environment.")
        groq_key = groq_key_env
    else:
        groq_key = st.text_input(
            "Enter your Groq API key",
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed"
        )

    st.divider()

    # Validation & Launch
    ready = dataset_files and glossary_file and groq_key

    if dataset_files:
        st.markdown("**Selected dataset files:**")
        for f in dataset_files:
            size_mb = round(f.size / (1024 * 1024), 2)
            st.markdown(f"- `{f.name}` — {size_mb} MB")

    if not ready:
        missing = []
        if not dataset_files: missing.append("dataset CSV(s)")
        if not glossary_file: missing.append("glossary CSV")
        if not groq_key: missing.append("Groq API key")
        st.info(f"⬆️ Still needed: {', '.join(missing)}")

    if st.button("🚀 Run Pipeline", disabled=not ready, use_container_width=True, type="primary"):
        # Save files to a temp session directory
        session_dir = Path(tempfile.mkdtemp(prefix="termmap_"))
        data_dir = session_dir / "data"
        results_dir = session_dir / "results"
        data_dir.mkdir()
        results_dir.mkdir()

        for f in dataset_files:
            (data_dir / f.name).write_bytes(f.read())

        glossary_path = data_dir / glossary_file.name
        glossary_path.write_bytes(glossary_file.read())

        st.session_state.session_dir = str(session_dir)
        st.session_state.glossary_name = glossary_file.name
        st.session_state.groq_key = groq_key
        st.session_state.phase = "pipeline"
        st.rerun()


# ─────────────────────────────────────────────
# PHASE 2 – PIPELINE PROGRESS
# ─────────────────────────────────────────────
def show_pipeline_phase():
    session_dir = Path(st.session_state.session_dir)
    data_dir = session_dir / "data"
    results_dir = session_dir / "results"
    glossary_name = st.session_state.glossary_name
    groq_key = st.session_state.groq_key

    st.markdown("""
    <div style="text-align:center; padding: 30px 0 10px 0;">
        <h1 style="font-size:2rem;">⚙️ Running Pipeline</h1>
        <p style="color:#9CA3AF;">Please wait while your data is being processed...</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("📋 Step 1: Profiling Columns",       "Analysing column statistics and generating column cards..."),
        ("🔢 Step 2: Generating Embeddings",    "Fitting TF-IDF vectorizer on column cards and glossary terms..."),
        ("🔍 Step 3: Retrieving Candidates",    "Computing cosine similarity to find top-3 glossary matches per column..."),
        ("🤖 Step 4: LLM Reranking",            "Calling Groq LLaMA to select the best term for each column..."),
    ]

    # Render all step cards first (greyed out)
    step_placeholders = []
    for label, desc in steps:
        ph = st.empty()
        step_placeholders.append(ph)
        ph.markdown(f"""
        <div style="background:#1F2937; border:1px solid #374151; border-radius:10px;
                    padding:16px 20px; margin-bottom:10px; color:#6B7280;">
            <b>{label}</b><br>
            <span style="font-size:0.85rem;">{desc}</span>
        </div>""", unsafe_allow_html=True)

    overall_bar = st.progress(0, text="Starting pipeline...")
    error_box = st.empty()

    def mark_running(i):
        label, desc = steps[i]
        step_placeholders[i].markdown(f"""
        <div style="background:#1F2937; border:1px solid #F59E0B; border-radius:10px;
                    padding:16px 20px; margin-bottom:10px; color:#FCD34D;">
            <b>⏳ {label}</b><br>
            <span style="font-size:0.85rem; color:#9CA3AF;">{desc}</span>
        </div>""", unsafe_allow_html=True)

    def mark_done(i, detail=""):
        label, desc = steps[i]
        step_placeholders[i].markdown(f"""
        <div style="background:#064E3B; border:1px solid #059669; border-radius:10px;
                    padding:16px 20px; margin-bottom:10px; color:#D1FAE5;">
            <b>✅ {label}</b><br>
            <span style="font-size:0.85rem; color:#6EE7B7;">{detail if detail else desc}</span>
        </div>""", unsafe_allow_html=True)

    def mark_error(i, err):
        label, _ = steps[i]
        step_placeholders[i].markdown(f"""
        <div style="background:#7F1D1D; border:1px solid #DC2626; border-radius:10px;
                    padding:16px 20px; margin-bottom:10px; color:#FCA5A5;">
            <b>❌ {label} — Failed</b><br>
            <span style="font-size:0.85rem;">{err}</span>
        </div>""", unsafe_allow_html=True)

    try:
        # Step 1 — Profile
        mark_running(0)
        overall_bar.progress(5, text="Profiling columns...")
        n_cards = run_profile(data_dir, results_dir, glossary_name)
        mark_done(0, f"Generated {n_cards} column cards.")
        overall_bar.progress(25, text="Profiling complete.")

        # Step 2 — Embed
        mark_running(1)
        overall_bar.progress(30, text="Generating TF-IDF embeddings...")
        card_shape, gloss_shape = run_embed(data_dir, results_dir, glossary_name)
        mark_done(1, f"Card embeddings: {card_shape} | Glossary embeddings: {gloss_shape}")
        overall_bar.progress(50, text="Embeddings complete.")

        # Step 3 — Retrieve
        mark_running(2)
        overall_bar.progress(55, text="Retrieving top-k candidates...")
        n_matches = run_retrieve(data_dir, results_dir, glossary_name)
        mark_done(2, f"Generated {n_matches} column-term match rows.")
        overall_bar.progress(70, text="Retrieval complete.")

        # Step 4 — LLM Rerank
        mark_running(3)
        overall_bar.progress(75, text="LLM reranking (this may take a few minutes)...")
        n_preds = run_llm_rerank(data_dir, results_dir, glossary_name, groq_key)
        mark_done(3, f"LLM predictions generated for {n_preds} columns.")
        overall_bar.progress(100, text="Pipeline complete! 🎉")

        st.session_state.pipeline_done = True

    except Exception as e:
        import traceback
        error_box.error(f"Pipeline failed: {e}\n\n{traceback.format_exc()}")
        if st.button("⬅️ Go Back and Try Again"):
            st.session_state.phase = "upload"
            st.rerun()
        return

    st.success("🎉 All steps complete! Loading the Review UI...")
    import time; time.sleep(1.5)
    st.session_state.phase = "review"
    st.rerun()


# ─────────────────────────────────────────────
# PHASE 3 – REVIEW UI
# ─────────────────────────────────────────────
def show_review_phase():
    session_dir = Path(st.session_state.session_dir)
    data_dir = session_dir / "data"
    results_dir = session_dir / "results"
    glossary_name = st.session_state.glossary_name

    cards = pd.read_csv(results_dir / "column_cards.csv")
    similarity = pd.read_csv(results_dir / "column_matches.csv")
    llm_preds = pd.read_csv(results_dir / "llm_predictions.csv")
    glossary = pd.read_csv(data_dir / glossary_name, dtype=str).fillna("")
    glossary_terms = glossary["TERM"].tolist()

    st.title("🧠 GenAI Business Term Mapping – Review UI")
    st.write("Review and approve business term mappings for each dataset column.")

    # Sidebar — reset button
    with st.sidebar:
        st.markdown("### Navigation")
        if st.button("🔄 Upload New Data", use_container_width=True):
            st.session_state.phase = "upload"
            st.session_state.pipeline_done = False
            st.session_state.session_dir = None
            st.rerun()

        tables = cards["table"].unique().tolist()
        selected_table = st.selectbox("Select Table", tables)
        table_columns = cards[cards["table"] == selected_table]["column"].tolist()
        selected_column = st.selectbox("Select Column", table_columns)

    card_row = cards[(cards["table"] == selected_table) & (cards["column"] == selected_column)].iloc[0]
    matches = similarity[(similarity["table"] == selected_table) & (similarity["column"] == selected_column)].sort_values("rank").head(3)
    llm_row = llm_preds[(llm_preds["table"] == selected_table) & (llm_preds["column"] == selected_column)].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Column Profile Card")
        st.code(card_row["card_text"], language="markdown")

    with col2:
        st.subheader("🔍 Top 3 Similarity Candidates (TF-IDF)")
        st.table(matches[["term", "score"]])

    st.markdown("""
    <div style="background-color:#1f2937; padding:20px; border-radius:10px; margin-top:20px;">
        <h3 style="color:#fbbf24;">🤖 LLM Suggested Term</h3>
    </div>""", unsafe_allow_html=True)

    llm_term = llm_row["llm_term"]
    llm_conf = llm_row["llm_confidence"]
    llm_reason = llm_row["llm_reason"]

    st.markdown(f"""
    <div style="background-color:#065F46; padding:18px; border-radius:10px; color:white;
                font-size:17px; font-weight:500; margin-bottom:10px; border:1px solid #0d8f64;
                margin-top:10px;">
        {llm_term}
        <br><span style="font-size:14px; opacity:0.8;">confidence: {llm_conf}</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color:#1F2937; padding:15px; border-radius:10px; color:#D1D5DB;
                font-size:14px; border:1px solid #374151;">
        <b>Reason:</b> {llm_reason}
    </div>""", unsafe_allow_html=True)

    st.subheader("📝 Human Review")

    default_idx = glossary_terms.index(llm_term) if llm_term in glossary_terms else 0
    selected_final = st.selectbox("Select Final Approved Term", glossary_terms, index=default_idx)

    if st.button("💾 Save Decision"):
        review_path = results_dir / "human_review.csv"
        new_row = pd.DataFrame([{"table": selected_table, "column": selected_column, "approved_term": selected_final}])
        if review_path.exists():
            old = pd.read_csv(review_path)
            pd.concat([old, new_row], ignore_index=True).to_csv(review_path, index=False)
        else:
            new_row.to_csv(review_path, index=False)
        st.success("✅ Saved successfully!")

    # Download results
    st.divider()
    st.subheader("⬇️ Download Results")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    with dl_col1:
        st.download_button("📥 Column Cards", data=open(results_dir / "column_cards.csv", "rb").read(), file_name="column_cards.csv", mime="text/csv", use_container_width=True)
    with dl_col2:
        st.download_button("📥 Column Matches", data=open(results_dir / "column_matches.csv", "rb").read(), file_name="column_matches.csv", mime="text/csv", use_container_width=True)
    with dl_col3:
        st.download_button("📥 LLM Predictions", data=open(results_dir / "llm_predictions.csv", "rb").read(), file_name="llm_predictions.csv", mime="text/csv", use_container_width=True)

    review_path = results_dir / "human_review.csv"
    if review_path.exists():
        st.download_button("📥 Human Review Decisions", data=open(review_path, "rb").read(), file_name="human_review.csv", mime="text/csv", use_container_width=True)


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
if st.session_state.phase == "upload":
    show_upload_phase()
elif st.session_state.phase == "pipeline":
    show_pipeline_phase()
elif st.session_state.phase == "review":
    show_review_phase()