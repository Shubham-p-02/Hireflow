# HireFlow — AI-Powered Resume Search

HireFlow indexes PDF resumes using hybrid search (BM25 + Pinecone vectors), fuses results with Reciprocal Rank Fusion, applies post-search filters, and optionally re-ranks candidates with a Gemini LLM evaluator.

---

## Setup

### 1. Create virtual environment and install dependencies

```bash
cd Hireflow
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file at the project root:

```
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=hireflow
GOOGLE_API_KEY=your_google_api_key
```

### 3. Add resumes

Place your PDF resumes in `data/resumes/`.

---

## Feature Walkthrough

Follow these steps in order. Each one builds on the previous.

### Feature 1 — Run the test suite

Verify everything works before touching any external APIs:

```bash
pytest tests/ -v
```

All tests are self-contained (Pinecone and Gemini are mocked). You should see
tests passing for: `test_filters`, `test_hybrid_indexer`, `test_re_ranker`, `test_ingestion`, `test_ir_metrics`, `test_rerank_eval`, `test_evaluator_ranking`.

---

### Feature 2 — Start the FastAPI backend

```bash
python start_backend.py
```

This starts the API at `http://localhost:8000`. Open `http://localhost:8000/docs` to see the Swagger UI.

**Check system status:**

```bash
curl http://localhost:8000/status
```

Response: `{"resumes_ready": false, "vector_store_ready": true, "hybrid_ready": false, "pinecone_vector_count": 0}`

---

### Feature 3 — Index resumes via the API

```bash
curl -X POST http://localhost:8000/index
```

This loads all PDFs from `data/resumes/`, embeds them, and upserts into Pinecone + builds BM25 index. Response:

```json
{"indexed": 50, "message": "Successfully indexed 50 resumes"}
```

Verify with `/status` again — `pinecone_vector_count` should now be > 0.

---

### Feature 4 — Search candidates via the API

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Senior Accountant with QuickBooks", "top_k": 5}'
```

Response includes three separate scores for each candidate:

```json
{
  "results": [
    {
      "candidate_id": "c_alice_smith",
      "name": "Alice Smith",
      "bm25_score": 0.8721,
      "vector_score": 0.7534,
      "combined_score": 0.0323,
      "skills": ["QuickBooks", "Excel", "Accounting"],
      "location": "New York, USA",
      "experience": 5
    }
  ],
  "total": 5
}
```

- **bm25_score** — normalized keyword match (0-1)
- **vector_score** — cosine similarity from Pinecone (0-1)
- **combined_score** — Reciprocal Rank Fusion across both lists

---

### Feature 5 — Launch the Streamlit web UI

Open a second terminal:

```bash
streamlit run streamlit/app.py
```

**Smart startup:** On launch, the app checks if Pinecone already has vectors. If yes, it only rebuilds the in-memory BM25 index (fast). If not, it does a full index. No redundant embedding work on restarts.

---

### Feature 6 — Upload new resumes via the UI

In the Streamlit app:
1. Use the **Add Resumes** panel on the left
2. Upload one or more PDF files
3. Click **Process & Index Resumes**

Each resume is parsed with Gemini to extract name, skills, location, and experience, then indexed into both BM25 and Pinecone.

---

### Feature 7 — Search with filters

In the **Search Candidates** panel:
1. Enter a **Job Title** (e.g. "Senior Accountant")
2. Enter a **Job Description** (e.g. "Looking for experienced accountant with tax expertise")
3. Enter **Required Skills** (one per line)
4. Optionally set a **Location Filter** and **Min. Experience**
5. Click **Find Candidates**

Post-search filters (`core/filters.py`) are applied after the hybrid search to narrow down by skills, location, and experience. Results show:
- BM25 Score, Vector Score, and Combined (RRF) as separate metrics
- Skills, experience, and location for each candidate
- Resume preview

---

### Feature 8 — AI evaluation and re-ranking

For the top 5 search results, the **ReRanker** (`core/re_ranker.py`) calls Gemini to produce:
- **Fit score** (0-100)
- **Strengths** — what makes this candidate a good match
- **Gaps** — where they fall short
- **Summary** — one-line assessment

These appear inside each candidate's expandable card. If Gemini is unavailable, a rule-based fallback scores based on skill overlap.

---

### Feature 9 — Memory and evaluation dashboard

In the Streamlit sidebar, click **Memory & Evaluation**:
- **Search Memory** — see total interactions, searches, and candidate views
- **Recent Search History** — last 5 queries
- **RAG Quality Evaluation** — enter a query and expected skills, then measure search quality using RAGAS metrics (answer relevancy, context precision, faithfulness, answer correctness)

---

### Feature 10 — Force re-index

If you add new resume PDFs to `data/resumes/` and want to refresh the index:
- **Sidebar** > click **Force Re-index Resumes**

This triggers a full BM25 + Pinecone rebuild.

---

### Feature 11 — IR ranking metrics (search & re-ranker quality)

RAGAS metrics (Feature 9) are QA/RAG metrics — they judge a generated answer
against context, not how good a *ranked list* of candidates is. To actually
measure the hybrid search and re-ranker, `core/evaluator.py` also exposes
standard information-retrieval metrics (Precision@K, Recall@K, MRR, NDCG@K,
MAP) computed against a heuristic skill-overlap relevance judgment. These
need no LLM calls, so they're fast and deterministic:

```python
from core.evaluator import RAGEvaluator

evaluator = RAGEvaluator(vector_store, hybrid_indexer)

# How good is the hybrid search's ranking?
ranking_metrics = evaluator.evaluate_ranking_quality(
    query="Senior Accountant with QuickBooks",
    expected_skills=["QuickBooks", "Tax Preparation", "Excel"],
    top_k=10,
)
print(ranking_metrics.to_dict())
# {'mrr': 0.83, 'map': 0.71, 'precision@3': 0.67, 'recall@10': 1.0, 'ndcg@10': 0.88, ...}

# Does LLM re-ranking actually improve on the raw hybrid ranking?
reranker_metrics = evaluator.evaluate_reranker_quality(
    query="Senior Accountant with QuickBooks",
    expected_skills=["QuickBooks", "Tax Preparation", "Excel"],
    top_k=5,
)
print(reranker_metrics.to_dict())
# {'fit_score_correlation': 0.74, 'ndcg_before': 0.62, 'ndcg_after': 0.81, 'ndcg_uplift': 0.19}
```

- `evaluate_ranking_quality` — Precision/Recall/NDCG@k, MRR, and MAP for the hybrid search, scored against every indexed candidate (not just the ones retrieved), so recall reflects candidates that were missed entirely.
- `evaluate_reranker_quality` — Spearman correlation between the re-ranker's `fit_score` and true relevance, plus NDCG uplift (post-rerank NDCG minus pre-rerank NDCG). A near-zero correlation or negative uplift means re-ranking isn't adding value over the hybrid search's own order.

See `core/ir_metrics.py` and `core/rerank_eval.py` for the underlying metric implementations, and `tests/test_ir_metrics.py`, `tests/test_rerank_eval.py`, `tests/test_evaluator_ranking.py` for examples.

### Labeled benchmark (real, multi-query)

The methods above take a single query and infer relevance from a skill-overlap
heuristic — useful for quick checks, but a single query is too noisy to draw
conclusions from. `benchmarks/benchmark.py` instead runs **9 hand-labeled
queries** (relevance judged by hand from the corpus's actual job-title
distribution — a real ground truth, not inferred from the resumes' own text)
through three configurations — BM25-only, hybrid (BM25 + Pinecone vector,
RRF-fused), and hybrid + LLM re-ranking — and reports the lift between them.

```bash
python benchmarks/benchmark.py   # requires PINECONE_API_KEY + GOOGLE_API_KEY in .env
```

Results from a real run (`benchmarks/RESULTS.md`, always regenerated by the
script, never hand-edited): adding vector search produced a **mixed** effect
on this corpus — MRR and NDCG@5 improved, but MAP, precision@10, and
recall@10 slightly *decreased* — and LLM re-ranking showed a weak positive
correlation with true relevance (0.21) but a net-neutral NDCG impact
(-0.01). Neither addition is a clear win here; that's the actual, honest
finding, not a failure of the benchmark.

---

## Project Structure

```
Hireflow/
├── api/
│   └── main.py              # FastAPI backend (POST /index, /search, GET /status)
├── core/
│   ├── hybrid_indexer.py    # BM25 + Pinecone vector search with RRF fusion
│   ├── vector_store.py      # Pinecone client wrapper
│   ├── ingestion.py         # PDF loading -> LangChain Documents
│   ├── parsing.py           # Gemini-based resume field extraction
│   ├── re_ranker.py         # LLM candidate evaluation (fit score 0-100)
│   ├── filters.py           # Post-search filtering (skills/location/experience)
│   ├── memory_rag.py        # Search history and interaction tracking
│   ├── search_router.py     # Shallow vs deep search routing
│   ├── evaluator.py         # RAGAS + IR ranking metrics
│   ├── ir_metrics.py        # Precision@K/Recall@K/MRR/NDCG@K (pure functions)
│   └── rerank_eval.py       # Re-ranker quality vs. relevance judgments
├── utils/
│   ├── schemas.py           # SearchQuery, Resume, CandidateEvaluation
│   ├── config.py            # .env configuration loader
│   ├── embeddings.py        # HuggingFace all-MiniLM-L6-v2
│   ├── utils.py             # Text processing, PDF loading, logging
│   └── multi_query.py       # LLM query expansion
├── streamlit/
│   └── app.py               # Web interface
├── tests/
│   ├── test_filters.py      # Filter function tests
│   ├── test_hybrid_indexer.py # RRF fusion and indexing tests
│   ├── test_re_ranker.py    # Evaluation and section parsing tests
│   ├── test_ingestion.py    # PDF loading tests
│   ├── test_ir_metrics.py   # Precision/Recall/NDCG/MRR/MAP tests
│   ├── test_rerank_eval.py  # Spearman correlation and NDCG uplift tests
│   └── test_evaluator_ranking.py # RAGEvaluator ranking-metrics methods
├── data/
│   └── resumes/             # Place PDF resumes here
├── main.py                  # CLI interface
├── start_backend.py         # uvicorn launcher
├── requirements.txt         # All dependencies
└── HireFlow_Architecture.md # Detailed architecture docs
```

---

## Technology Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (gemini-2.5-flash-lite) |
| Vector DB | Pinecone (serverless, cosine, 384 dims) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Keyword Search | rank_bm25 (BM25Okapi) |
| Score Fusion | Reciprocal Rank Fusion (k=60) |
| Orchestration | LangChain |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| RAG Evaluation | RAGAS |
| Testing | pytest |
