# Production-Ready RAG System

A robust, modular, and production-ready Retrieval-Augmented Generation (RAG) backend. This project goes beyond basic prototypes by implementing advanced retrieval techniques (Hybrid Search + Reciprocal Rank Fusion), re-ranking (CrossEncoder), and an automated evaluation pipeline (LLM-as-a-Judge using Ragas and YandexGPT).

## 🌟 Key Features

*   **Multi-Format Document Ingestion:** Supports loading context from PDF files, Markdown documents, and Web URLs.
*   **Vector Content Storage:** Uses local ChromaDB combined with standard `SentenceTransformers` embeddings.
*   **Hybrid Search (Lexical + Semantic):** Combines standard BM25 keyword search with dense vector search to retrieve documents accurately even using specific IDs, acronyms, or misspellings.
*   **Reciprocal Rank Fusion (RRF):** Custom robust implementation to mathematically merge and normalize search results from BM25 and Vector retrievers.
*   **Cross-Encoder Re-Ranking:** Implements a second-stage retrieval pipeline using MS MARCO MiniLM cross-encoder to accurately score and re-order the retrieved chunks for maximum relevance to the user's query.
*   **Citation & Prompt Management:** Strict system prompts managed externally (`config/prompts.yaml`) forcing the LLM to ground its answers exclusively in retrieved contexts and cite sources.
*   **Automated Evaluation Pipeline (CI/CD Ready):** Includes a `golden_dataset.json` and a script (`evaluate.py`) that utilizes the **Ragas** framework to evaluate the **Faithfulness** of the system using YandexGPT, natively returning exit codes suitable for GitHub Actions.

## 🛠️ Tech Stack
*   **Frameworks:** LangChain, HuggingFace Transformers
*   **Databases:** ChromaDB
*   **Algorithms:** BM25 (Rank-BM25), RRF, CrossEncoder
*   **Evaluation:** Ragas, YandexGPT API
*   **CI/CD:** GitHub Actions

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd RAG
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your Yandex Cloud credentials (required for evaluation):
```env
YC_API_KEY=your_yandex_api_key
YC_FOLDER_ID=your_yandex_folder_id
```

### 3. Usage

The project backend is controlled via `main.py`. You can import the functions to process documents or query the system:

```python
from main import ingest_data, query_system

# 1. Index a document
ingest_data("data/company_policy.pdf", "pdf")

# 2. Query the system natively (uses Hybrid Search + Re-ranking)
query_system("What is the vacation policy for remote employees?")
```

### 4. Running the Evaluation

To check the system's performance, run the evaluation script against the Golden Dataset:
```bash
python evaluate.py
```
*Note: This will use YandexGPT to score the Faithfulness metric and determine if the answers meet the 0.85 threshold.*

## 📈 System Architecture Pipeline
1. **Load -> Chunk -> Embed -> ChromaDB**
2. **User Query -> BM25 Retriever & Vector Retriever -> RRF Normalization**
3. **Top 10 Chunks -> CrossEncoder Re-Ranking -> Top 3 Chunks**
4. **Top 3 Chunks + Prompt -> LLM Generation -> Answer**
