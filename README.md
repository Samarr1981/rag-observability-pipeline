# RAG Observability Pipeline

A production-adjacent RAG pipeline built on the original [RAG research paper](https://arxiv.org/abs/2005.11401) with hybrid retrieval, cross-encoder reranking, page-level citations, and full LangSmith observability and evaluation.

---

## What This Does

Answers questions about a research paper PDF with cited, grounded responses. Every run is fully traced in LangSmith — you can see exactly which chunks were retrieved, what prompt was assembled, token usage, latency, and cost per question. A custom LLM-as-judge evaluator scores answer correctness automatically.

---

## Pipeline Architecture

```
rag_paper.pdf
      ↓
PyPDFLoader              — loads PDF page by page
      ↓
SemanticChunker          — splits by meaning using OpenAI embeddings
      ↓
FAISS + BM25             — hybrid search via EnsembleRetriever (Reciprocal Rank Fusion)
      ↓
CrossEncoder reranker    — ms-marco-MiniLM-L-6-v2 rescores top candidates
      ↓
gpt-4o-mini              — generates answer with page-level citations
      ↓
LangSmith                — traces every step + evaluates correctness
```

---

## Stack

- **LangChain** — pipeline orchestration
- **LangSmith** — tracing, monitoring, and evaluation
- **OpenAI** — embeddings (`text-embedding-ada-002`) and generation (`gpt-4o-mini`)
- **FAISS** — vector store for semantic search
- **BM25** — keyword-based retrieval
- **sentence-transformers** — cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **FAISS + BM25 via EnsembleRetriever** — hybrid search with RRF merging

---

## Key Design Decisions

**Semantic Chunking over Recursive Character Splitting**
SemanticChunker uses embedding similarity to find natural topic boundaries instead of splitting by character count. This preserves context better for research papers with dense, structured content.

**Hybrid Search**
Pure vector search misses exact keyword matches (e.g. "DPR", "TriviaQA"). Pure BM25 misses semantic similarity. EnsembleRetriever combines both using Reciprocal Rank Fusion so a chunk that ranks well in either list floats to the top.

**Cross-Encoder Reranking**
Bi-encoders (FAISS embeddings) score query and document independently. A cross-encoder sees them jointly, producing more accurate relevance scores. Reranking the top 6 candidates down to 4 consistently improves answer quality.

**LangSmith Evaluation**
Not just "does it answer" but "is the answer correct". A custom `CorrectnessEvaluator` uses gpt-4o-mini as a judge to score each answer against a reference answer. Results are tracked in LangSmith Datasets and Experiments.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Samarr1981/rag-observability-pipeline.git
cd rag-observability-pipeline
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install langchain langchain-openai langchain-community langchain-classic \
    langchain-experimental langchain-text-splitters langsmith \
    faiss-cpu rank-bm25 sentence-transformers pypdf python-dotenv
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=doc-qa-agent
OPENAI_API_KEY=your_openai_key
HF_TOKEN=your_huggingface_token
```

Get your LangSmith API key at [smith.langchain.com](https://smith.langchain.com).

**5. Add your PDF**

Download the RAG paper and save it as `rag_paper.pdf` in the project root:
```
https://arxiv.org/pdf/2005.11401
```

Or swap in any PDF you want to query.

**6. Run**
```bash
python3 agent.py
```

---

## What You'll See

**Terminal output** — 5 questions answered with page citations:
```
Q: What retrieval method does the RAG paper use?
A: The RAG paper employs a learned dense retrieval mechanism...
   [Source: Page 6]
```

**LangSmith trace** — full run tree showing:
- EnsembleRetriever (BM25 + VectorStore running in parallel)
- CrossEncoder reranking step
- ChatOpenAI node with exact prompt, token count, and cost

**LangSmith evaluation** — correctness scores across 3 experiment runs in Datasets and Experiments.

---

## Project Structure

```
rag-observability-pipeline/
├── agent.py        — main pipeline
├── rag_paper.pdf   — document to query (not committed)
├── .env            — API keys (not committed)
├── .gitignore
└── README.md
```

---

## What's Missing for Production

This is a learning project. Moving to production would require:

- Managed vector DB (Pinecone, Qdrant, or Weaviate) instead of local FAISS
- Agentic retrieval loop — query reformulation if the first retrieval isn't sufficient
- Access control and audit logging
- RAGAS metrics (faithfulness, answer relevancy, context precision) for more rigorous evaluation

---

## Author

Samar Passey — [GitHub](https://github.com/Samarr1981) · [LinkedIn](https://www.linkedin.com/in/samar-passey/)