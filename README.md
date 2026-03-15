# 🧠 Automated Business Term Identification

A GenAI-powered pipeline that automatically maps dataset columns to standardized business glossary terms using a hybrid approach: **TF-IDF similarity retrieval** + **LLM re-ranking** + **human review UI**.

---

## 📌 Overview

When organizations manage multiple datasets (e.g., from e-commerce platforms like Amazon, Walmart, Shopee), column names are often inconsistent or ambiguous. This project solves that by automatically identifying the correct business term (e.g., `init_price` → `Initial Price`) from a curated glossary.

The pipeline follows three stages:

```
Raw CSV Data
     │
     ▼
[1] Profile Columns      →  column_cards.csv / .json
     │
     ▼
[2] Embed + Retrieve     →  TF-IDF similarity scores
     │
     ▼
[3] LLM Re-rank          →  llm_predictions.csv  (via Groq / LLaMA 3.1)
     │
     ▼
[4] Human Review UI      →  Streamlit app  (approve / override)
     │
     ▼
[5] Evaluate             →  Top-1 / Top-3 accuracy, F1 score
```

---

## 🗂️ Project Structure

```
automated_business_term_identification/
├── data/
│   ├── glossary.csv              # Business term definitions & synonyms
│   ├── amazon-products.csv
│   ├── lazada-products.csv
│   ├── shein-products.csv
│   ├── shopee-products.csv
│   └── walmart-products.csv
│
├── src/
│   ├── profile.py                # Step 1 – Generate column profile cards
│   ├── embed.py                  # Step 2 – TF-IDF vectorization
│   ├── retrieve.py               # Step 3 – Cosine similarity retrieval
│   ├── llm_rerank.py             # Step 4 – LLM re-ranking via Groq
│   ├── evaluate.py               # Step 5 – TF-IDF baseline accuracy
│   ├── evaluate_llm.py           # Step 5 – LLM accuracy (with synonym matching)
│   └── app.py                    # Step 6 – Streamlit review UI
│
├── results/
│   ├── column_cards.csv/.json    # Column profile cards
│   ├── column_matches.csv        # Top-3 TF-IDF candidate matches
│   ├── llm_predictions.csv       # LLM term predictions
│   └── gold_labels.csv           # Ground truth labels for evaluation
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/srivatsavdevarakonda/automated_business_term_identification.git
cd automated_business_term_identification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your Groq API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at [console.groq.com](https://console.groq.com)

---

## 🚀 Running the Pipeline

Run each step in order from the project root:

```bash
# Step 1 – Profile all CSV files in data/ and generate column cards
python src/profile.py

# Step 2 – Compute TF-IDF embeddings for columns and glossary terms
python src/embed.py

# Step 3 – Retrieve top-3 matching glossary terms per column
python src/retrieve.py

# Step 4 – Use LLM to re-rank and select the best term
python src/llm_rerank.py

# Step 5 – Evaluate results
python src/evaluate.py        # TF-IDF baseline (Top-1 accuracy)
python src/evaluate_llm.py    # LLM accuracy with synonym matching

# Step 6 – Launch the human review UI
streamlit run src/app.py
```

---

## 🖥️ Review UI

The Streamlit app provides a human-in-the-loop interface to review and approve predictions.

For each dataset column, you can see:

- 📄 **Column Profile Card** — data type, row count, null %, distinct values, sample values, and auto-detected hints (e.g., `date_like`, `numeric_dtype`)
- 🔍 **Top-3 TF-IDF Candidates** — similarity scores from the retrieval step
- 🤖 **LLM Suggested Term** — the model's best pick with confidence score and reasoning
- ✅ **Human Review** — override or approve the suggested term and save the decision

```bash
streamlit run src/app.py
```

---

## 🔬 How It Works

### Column Profiling (`profile.py`)
Each CSV column is converted into a structured **Column Card** containing its data type, null percentage, distinct value count, sample values, and auto-detected semantic hints.

### TF-IDF Embedding (`embed.py`)
Column cards and glossary terms are vectorized using a **character n-gram TF-IDF** (3–5 grams). Column names are repeated 3× to boost their weight during similarity matching.

### Cosine Retrieval (`retrieve.py`)
Cosine similarity is computed between every column embedding and all glossary term embeddings. The top-3 candidates are saved for each column.

### LLM Re-ranking (`llm_rerank.py`)
A rule-based pass first handles simple, high-confidence mappings (e.g., columns containing `price`, `brand`, `rating`). For the rest, the top-3 candidates are sent to **LLaMA 3.1 (8B)** via the Groq API with a structured prompt, returning the best term, a confidence score (0–1), and a short reasoning explanation.

### Evaluation (`evaluate_llm.py`)
Predictions are compared against gold labels using exact match and **synonym-aware matching** (checking the glossary's synonym list). Reports Top-1 accuracy, Top-3 accuracy, Precision, Recall, and F1 score.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Human review UI |
| `pandas` | Data manipulation |
| `numpy` | Vector operations |
| `scikit-learn` | TF-IDF vectorization |
| `scipy` | (Similarity utilities) |
| `groq` | LLM inference via Groq API |
| `python-dotenv` | API key management |

---

## 📊 Data

The `data/` folder includes product datasets scraped from five e-commerce platforms (Amazon, Walmart, Shopee, Lazada, Shein) and a hand-curated `glossary.csv` with business terms, definitions, and synonyms.

---

## 📝 License

This project is for educational and research purposes.
