import streamlit as st
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

# Load data
cards = pd.read_csv(RESULTS_DIR / "column_cards.csv")
similarity = pd.read_csv(RESULTS_DIR / "column_matches.csv")
llm_preds = pd.read_csv(RESULTS_DIR / "llm_predictions.csv")

# Load glossary for dropdown list
glossary = pd.read_csv(DATA_DIR / "glossary.csv")
glossary_terms = glossary["TERM"].tolist()

st.set_page_config(page_title="GenAI Business Term Mapper", layout="wide")

st.title("🧠 GenAI Business Term Mapping – Review UI")
st.write("Review and approve business term mappings for each dataset column.")

# Select table
tables = cards["table"].unique().tolist()
selected_table = st.sidebar.selectbox("Select Table", tables)

# Filter columns for the selected table
table_columns = cards[cards["table"] == selected_table]["column"].tolist()
selected_column = st.sidebar.selectbox("Select Column", table_columns)

# Get column card text
card_row = cards[
    (cards["table"] == selected_table) &
    (cards["column"] == selected_column)
].iloc[0]

# Get similarity matches (top 3)
matches = similarity[
    (similarity["table"] == selected_table) &
    (similarity["column"] == selected_column)
].sort_values("rank").head(3)

# Get LLM output
llm_row = llm_preds[
    (llm_preds["table"] == selected_table) &
    (llm_preds["column"] == selected_column)
].iloc[0]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Column Profile Card")
    st.code(card_row["card_text"], language="markdown")

with col2:
    st.subheader("🔍 Top 3 Similarity Candidates (TF-IDF)")
    st.table(matches[["term", "score"]])

# --- LLM Suggested Term Card ---
st.markdown("""
<div style="background-color:#1f2937; padding:20px; border-radius:10px; margin-top:20px;">
    <h3 style="color:#fbbf24;">🤖 LLM Suggested Terms are seen below!!!</h3>
</div>
""", unsafe_allow_html=True)

llm_term = llm_row["llm_term"]
llm_conf = llm_row["llm_confidence"]
llm_reason = llm_row["llm_reason"]

# Green card with LLM suggestion
st.markdown(
    f"""
    <div style="
        background-color:#065F46;   /* Dark green */
        padding:18px;
        border-radius:10px;
        color:white;
        font-size:17px;
        font-weight:500;
        margin-bottom:10px;
        border:1px solid #0d8f64;
    ">
        {llm_term}  
        <br>
        <span style="font-size:14px; opacity:0.8;">confidence: {llm_conf}</span>
    </div>
    """,
    unsafe_allow_html=True
)


# Grey card showing the reason
st.markdown(
    f"""
    <div style="
        background-color:#1F2937;   /* Dark slate */
        padding:15px;
        border-radius:10px;
        color:#D1D5DB;
        font-size:14px;
        border:1px solid #374151;
    ">
        <b>Reason:</b> {llm_reason}
    </div>
    """,
    unsafe_allow_html=True
)


st.subheader("📝 Human Review")

selected_final = st.selectbox(
    "Select Final Approved Term",
    glossary_terms,
    index=glossary_terms.index(llm_row["llm_term"])
    if llm_row["llm_term"] in glossary_terms else 0
)

if st.button("Save Decision"):
    review_path = RESULTS_DIR / "human_review.csv"
    new_row = pd.DataFrame([{
        "table": selected_table,
        "column": selected_column,
        "approved_term": selected_final
    }])

    # Append or create new
    if review_path.exists():
        old = pd.read_csv(review_path)
        merged = pd.concat([old, new_row], ignore_index=True)
        merged.to_csv(review_path, index=False)
    else:
        new_row.to_csv(review_path, index=False)

    st.success("Saved successfully!")
