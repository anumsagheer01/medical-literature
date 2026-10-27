<div align="center">

# Medical Literature RAG

### Search medical research · Generate cited answers · Evaluate retrieval quality

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-PubMed-127B75?style=for-the-badge)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_Search-19647E?style=for-the-badge)
![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-8E44AD?style=for-the-badge)
![Hugging%20Face](https://img.shields.io/badge/Hugging_Face-Live_Demo-F5A623?style=for-the-badge&logo=huggingface&logoColor=white)

**[View the live demo](https://huggingface.co/spaces/anumsagheer/medical-literature)**

</div>

---

## Project Overview & Key Results

Built a **medical literature RAG system** that retrieves PubMed research and generates cited answers to clinical questions, with confidence scores and uncertainty flags. The current collection covers **diabetes, depression, and cardiovascular disease**.

| Area | Approach & Technologies | Key Results |
|---|---|---|
| **Medical Literature Retrieval** | PubMed / NCBI Entrez, OpenAI embeddings, Pinecone | **3,775 abstracts** indexed across three disease areas |
| **Retrieval & Reranking** | Sentence-window chunking, Cohere Rerank v3 | Top-result score increased from **0.640 to 0.990** in a documented comparison |
| **RAG Evaluation** | GPT-4o-mini, RAGAS, 47-question evaluation set | **0.906 faithfulness**, **0.749 answer relevancy**; context recall **0.495** |
| **LLM Inference Benchmark** | Llama 3.1 8B, 4-bit quantization, T4 GPU | Average latency fell from **67.9s to 7.7s** in a separate benchmark |
| **Application & Monitoring** | FastAPI, Streamlit, LangSmith, Hugging Face Spaces | Deployed demo with citations, confidence scores, and uncertainty flags |

*Evaluation is based on a limited set of PubMed abstracts and test questions, not a clinical deployment. Reranker scores and RAGAS metrics measure different aspects of performance; the Llama benchmark is separate from the GPT-4o-mini answer pipeline.*

## The Problem

Medical research can be difficult to search and synthesize. This project explores whether retrieval and reranking can help people find relevant PubMed evidence and get a clear answer with citations. When evidence is limited, the interface is designed to flag uncertainty rather than present an unsupported conclusion.

## What it covers


| Disease Area | Papers Indexed |
|---|---|
| Type 2 Diabetes | 1,332 |
| Major Depression | 1,385 |
| Cardiovascular Disease | 1,058 |
| **Total** | **3,775** |

All papers sourced from PubMed via the NCBI Entrez API. Each paper stored with PMID, title, authors, journal, year, and disease area as metadata.


## Architecture
```
PubMed API (NCBI Entrez)
        ↓
Document Ingestion Pipeline
        ↓
Chunking Layer - sentence window strategy (winner of 3-way benchmark)
        ↓
OpenAI text-embedding-3-small (1536 dimensions)
        ↓
Pinecone Vector Index - dual namespace (abstracts + sentence_window)
        ↓
Query → Pinecone retrieval (top 20) → Cohere Reranker (top 5)
        ↓
Context Window Manager - dynamic token budgeting
        ↓
Versioned Prompt Framework (v3 - chain of thought + structured JSON)
        ↓
GPT-4o-mini - answer + citations + confidence score + uncertainty flags
        ↓
RAGAS Evaluation Suite - faithfulness, relevancy, precision, recall
        ↓
LangSmith Tracing - observability, cost, latency per query
        ↓
FastAPI Backend + Streamlit UI
        ↓
Deployed on HuggingFace Spaces
```

## How it works

**1. Data ingestion**
The pipeline fetches PubMed abstracts through the NCBI Entrez API using disease-specific queries. Each paper is cleaned and saved with metadata used to generate citations.

**2. Chunking**
I compared fixed-size (512-token), semantic, and sentence-window chunking. Sentence-window chunking had the highest average similarity score (**0.610**, compared with **0.585** for fixed-size) and was selected for the RAG pipeline.

**3. Indexing**
Papers are embedded using OpenAI's text-embedding-3-small (1536 dimensions, cosine similarity) and stored in Pinecone with two namespaces — one for raw abstracts and one for sentence-window chunks. Metadata filtering by disease area, year, and journal is supported at query time.

**4. Retrieval + reranking**
A user query is embedded and matched against the `sentence_window` namespace. Pinecone returns top 20 candidates by cosine similarity. Cohere's reranker then re-scores all 20 against the actual query intent, not just vector similarity, and returns the top 5. For one documented query, the top-result score was **0.640** before reranking and **0.990** after Cohere reranking. These scores come from different scoring stages.

**5. Prompt framework**
Three prompt versions were built and tested. Version 3 uses an explicit role, grounding rules, structured JSON output (answer, citations, confidence score, and uncertainty flags), and guardrails for off-topic questions.

**6. Evaluation**
A 47-question clinical test set was built across all three disease areas with gold-standard answers. The evaluation reports four RAGAS metrics: faithfulness, answer relevancy, context precision, and context recall. LangSmith tracing supports analysis of latency, token use, and failure patterns.


## Results

### RAGAS Evaluation (47-question test set)

| Metric | Score | What it means |
|---|---|---|
| Faithfulness | **0.906** | RAGAS faithfulness score for grounding against retrieved context |
| Answer Relevancy | **0.749** | RAGAS answer relevancy score |
| Context Precision | **0.743** | RAGAS context precision score |
| Context Recall | **0.495** | RAGAS context recall score on the test set |

Context recall was the lowest-scoring RAGAS metric. The index uses abstracts rather than full articles, which limits the evidence available for retrieval. Indexing full-text PMC articles is a planned next step.

### Chunking strategy benchmark

| Strategy | Avg Similarity | Precision |
|---|---|---|
| Fixed-size (512 tokens) | 0.585 | 1.00 |
| Semantic (sentence boundary) | 0.587 | 1.00 |
| **Sentence window (winner)** | **0.610** | **1.00** |

### Reranker impact

| Stage | Top result score |
|---|---|
| Raw Pinecone retrieval | 0.640 |
| After Cohere reranking | 0.990 |

In the documented query example, reranking surfaced papers about treatment goals and medication management instead of more general diabetes papers. These scores are reranker outputs, not a direct measure of answer accuracy.

### Separate inference benchmark (Llama 3.1 8B on T4 GPU)

| Model | AVG | P50 | P90 |
|---|---|---|---|
| Float16 full precision | 67,914ms | 67,809ms | 68,564ms |
| **4-bit quantized** | **7,678ms** | **7,963ms** | **8,232ms** |
| **Improvement** | **+89%** | **+88%** | **+88%** |

In this separate Llama 3.1 8B benchmark, 4-bit quantization reduced average inference latency by about 89%. This test does not measure the latency of the deployed GPT-4o-mini RAG pipeline.

### LangSmith failure patterns identified

Three failure patterns were documented through tracing:
1. **Recently approved drugs** - tirzepatide returned 0.5 confidence, correctly flagged insufficient evidence
2. **Genetic specificity** - cardiovascular genetics queries returned 0.6 confidence, flagged missing variant details
3. **Direct drug comparisons** - metformin vs SGLT2 flagged "no direct comparison in literature"

In all three cases the system flagged low confidence rather than hallucinating, which is the correct behavior for healthcare AI.

## Tech stack

| Layer | Technology |
|---|---|
| Data source | PubMed via NCBI Entrez API (Biopython) |
| Vector database | Pinecone (serverless, cosine, 1536d) |
| Embeddings | OpenAI text-embedding-3-small |
| Reranker | Cohere Rerank v3 |
| LLM | OpenAI GPT-4o-mini |
| Fine-tuning | Llama 3.1 8B via Unsloth + LoRA on MedQA |
| Evaluation | RAGAS |
| Observability | LangSmith |
| Inference benchmark | HuggingFace Transformers + BitsAndBytes |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | HuggingFace Spaces |
| Language | Python 3.11 |


## Repository guide
```
medical-literature/
│
├── app.py                          # Streamlit UI, main entry point
├── main.py                         # FastAPI backend
├── requirements.txt
├── .env                            # API keys (not committed)
│
├── data/
│   └── papers.json                 # 3,775 PubMed abstracts
│
├── eval/
│   ├── eval_set.json               # 47-question gold standard test set
│   ├── ragas_results.json          # RAGAS evaluation scores
│   └── answers_cache.json          # Cached pipeline outputs
│
├── prompts/
│   ├── system_v1.txt               # Basic role definition
│   ├── system_v2.txt               # + guardrails + uncertainty flagging
│   ├── system_v3.txt               # + chain of thought + JSON output
│   ├── user_template.txt           # Query + context template
│   └── changelog.md                # What changed between versions
│
└── scripts/
    ├── fetch_pubmed.py             # PubMed ingestion pipeline
    ├── index_pinecone.py           # Embedding + Pinecone indexing
    ├── chunking_experiment.py      # 3-strategy chunking benchmark
    ├── reranker.py                 # Cohere reranker comparison
    ├── rerank_test.py              # Reranker quality test
    ├── prompt_framework.py         # v1/v2/v3 prompt comparison
    ├── rag_pipeline.py             # Basic RAG pipeline
    ├── run_ragas.py                # RAGAS evaluation suite
    └── langsmith_trace.py          # LangSmith observability setup
```

## Run it locally

**1. Clone the repo**
```bash
git clone https://github.com/anumsagheer01/medical-literature
cd medical-literature
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key
COHERE_API_KEY=your_key
NCBI_API_KEY=your_key
NCBI_EMAIL=your@email.com
```

**5. Run the app**

If your Pinecone index is already configured and populated, you can start the application without rerunning ingestion:
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.


## What's next

- Expand corpus to additional disease areas (oncology, neurology, infectious disease)
- Index full-text PMC articles to improve context recall from 0.495
- Add hybrid search combining dense vectors with BM25 keyword search
- Query understanding layer to route different question types to different retrieval strategies
- Run RAGAS evaluations whenever the corpus changes
