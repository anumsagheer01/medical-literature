## Medical Literature Search
> A production RAG system that answers clinical questions from peer-reviewed research, with citations, confidence scores, and uncertainty flags.

**Live Demo:** https://huggingface.co/spaces/anumsagheer/medical-literature

> Currently covers three disease areas: **diabetes, depression, and cardiovascular disease.** More categories are actively being added.

### What is this?

Researchers and clinicians can't keep up with the 1 million+ new PubMed papers published every year. Keyword search returns 500 results nobody reads. This system lets you ask a plain English clinical question and get back a direct, cited answer drawn from real peer-reviewed research, along with a confidence score and explicit flags when the evidence is limited or uncertain.

This is not a chatbot that makes things up. Every claim in every answer traces back to a specific paper. If the corpus doesn't have enough evidence, the system says so, with 0% confidence and a clear explanation. 

### What it covers

| Disease Area | Papers Indexed |
|---|---|
| Type 2 Diabetes | 1,332 |
| Major Depression | 1,385 |
| Cardiovascular Disease | 1,058 |
| **Total** | **3,775** |

All papers sourced from PubMed via the NCBI Entrez API. Each paper stored with PMID, title, authors, journal, year, and disease area as metadata.


### Architecture
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

### How it works

**1. Data ingestion**
The pipeline fetches PubMed abstracts using the NCBI Entrez API with disease-specific queries. Each paper is cleaned and stored with rich metadata that later becomes the citation shown to users.

**2. Chunking**
Three strategies were benchmarked, fixed-size (512 tokens), semantic (sentence boundary), and sentence window (embed sentences, return surrounding context). Sentence window won with the highest average similarity score (0.610 vs 0.585 for fixed-size) and was selected as the production strategy.

**3. Indexing**
Papers are embedded using OpenAI's text-embedding-3-small (1536 dimensions, cosine similarity) and stored in Pinecone with two namespaces — one for raw abstracts and one for sentence-window chunks. Metadata filtering by disease area, year, and journal is supported at query time.

**4. Retrieval + reranking**
User query is embedded and matched against the sentence_window namespace. Pinecone returns top 20 candidates by cosine similarity. Cohere's reranker then re-scores all 20 against the actual query intent, not just vector similarity, and returns the top 5. This two-stage approach dramatically improves result quality: raw Pinecone top score was 0.640, after Cohere reranking it jumped to 0.990 on the same query.

**5. Prompt framework**
Three prompt versions were built and tested. Version 3 (production) includes: explicit role definition, strict grounding rules, chain-of-thought reasoning (the model reasons step by step before answering), structured JSON output with answer/citations/confidence_score/uncertainty_flags, and medical domain guardrails that reject off-topic questions.

**6. Evaluation**
A 47-question clinical test set was built across all three disease areas with gold-standard answers. RAGAS runs four automatic metrics on every architecture change, faithfulness, answer relevancy, context precision, and context recall. All LLM calls are traced in LangSmith for latency, token usage, and failure pattern analysis.


### Performance metrics

### RAGAS Evaluation (47-question test set)

| Metric | Score | What it means |
|---|---|---|
| Faithfulness | **0.906** | 90.6% of answer claims are grounded in retrieved papers, very low hallucination rate |
| Answer Relevancy | **0.749** | Answers directly address the question asked 74.9% of the time |
| Context Precision | **0.743** | Retrieved chunks are genuinely relevant to the query 74.3% of the time |
| Context Recall | **0.495** | System captures about half the available evidence, limited by abstract-only indexing |

Context recall is the weakest metric and that is expected as this indexs abstracts only, not full paper text. The fix is indexing full PMC articles, which is a planned next step.

#### Chunking strategy benchmark

| Strategy | Avg Similarity | Precision |
|---|---|---|
| Fixed-size (512 tokens) | 0.585 | 1.00 |
| Semantic (sentence boundary) | 0.587 | 1.00 |
| **Sentence window (winner)** | **0.610** | **1.00** |

#### Reranker impact

| Stage | Top result score |
|---|---|
| Raw Pinecone retrieval | 0.640 |
| After Cohere reranking | 0.990 |

The reranker completely changed which papers surfaced, generic diabetes papers were replaced by papers specifically about treatment goals and medication management.

#### Inference benchmark (Llama 3.1 8B on T4 GPU)

| Model | AVG | P50 | P90 |
|---|---|---|---|
| Float16 full precision | 67,914ms | 67,809ms | 68,564ms |
| **4-bit quantized** | **7,678ms** | **7,963ms** | **8,232ms** |
| **Improvement** | **+89%** | **+88%** | **+88%** |

4-bit quantization reduced inference latency by 88% with negligible quality loss, demonstrating production viability of quantized models for real-time medical QA.

#### LangSmith failure patterns identified

Three failure patterns were documented through tracing:
1. **Recently approved drugs** - tirzepatide returned 0.5 confidence, correctly flagged insufficient evidence
2. **Genetic specificity** - cardiovascular genetics queries returned 0.6 confidence, flagged missing variant details
3. **Direct drug comparisons** - metformin vs SGLT2 flagged "no direct comparison in literature"

In all three cases the system flagged low confidence rather than hallucinating, which is the correct behavior for healthcare AI.

### Tech stack

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


### Folder structure
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

### How to run locally

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

**5. The data is already indexed in Pinecone** you don't need to re-run the ingestion scripts. Just run the app:
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.


### What's next

- Expand corpus to additional disease areas (oncology, neurology, infectious disease)
- Index full-text PMC articles to improve context recall from 0.495
- Add hybrid search combining dense vectors with BM25 keyword search
- Query understanding layer to route different question types to different retrieval strategies
- Evaluation-driven development run RAGAS on every new corpus addition
