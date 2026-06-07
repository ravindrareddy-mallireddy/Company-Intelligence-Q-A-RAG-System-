# Company Intelligence Q&A — RAG System

A Retrieval-Augmented Generation (RAG) pipeline for querying 13,000+ UK companies extracted from the Companies House API.

## Architecture

```
User Query
    ↓
Embed (sentence-transformers: all-MiniLM-L6-v2)
    ↓
Retrieve top-k companies (ChromaDB vector database)
    ↓
Generate answer (Groq API — Llama 3 8B)
    ↓
Response + similarity scores + latency metrics
```

## Stack

| Component | Tool |
|-----------|------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| LLM | Groq API (Llama 3 8B) |
| Data Source | Companies House API (via Company Intelligence Extractor) |

## Setup

1. Install dependencies:
```bash
pip install chromadb groq sentence-transformers python-dotenv matplotlib pandas numpy
```

2. Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

3. Ensure these files are in the project folder:
- `companies_final.csv` — cleaned company data
- `embeddings.npy` — pre-computed sentence-transformer embeddings

4. Run the notebook:
```bash
jupyter notebook company_intelligence_rag.ipynb
```

## Features

- Semantic search over 13,000+ UK companies using cosine similarity
- Natural language answer generation via Llama 3
- Retrieval quality evaluation (latency, similarity scores)
- Cluster distribution visualisation
- Interactive Q&A loop
- Error handling and input validation

## Example Queries

- "Find software companies based in London"
- "Which companies work in cybersecurity?"
- "Show me fintech or financial services companies"
- "Find AI or machine learning companies in Manchester"

## Project Structure

```
company-rag/
├── company_intelligence_rag.ipynb  # Main RAG pipeline
├── companies_final.csv             # Company data
├── embeddings.npy                  # Pre-computed embeddings
├── rag_evaluation.png              # Generated evaluation charts
├── .env                            # API keys (not committed to git)
└── README.md
```

## Related Project

This project extends the [[Company Intelligence Extractor](https://github.com/ravindrareddy-mallireddy/Company-Intelligence-Extractor)] which extracted and clustered the underlying company data using the Companies House API, HDBSCAN, and UMAP.
