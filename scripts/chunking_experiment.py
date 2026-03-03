import json
import os
import time
import re
import nltk
import tiktoken
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

EMBEDDING_MODEL = "text-embedding-3-small"
enc = tiktoken.get_encoding("cl100k_base")

# ─── 3 CHUNKING STRATEGIES ───────────────────────────────────────────

def chunk_fixed(paper, chunk_size=512, overlap=50):
    """Strategy 1: Fixed token size with overlap"""
    tokens = enc.encode(paper["abstract"])
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = enc.decode(tokens[start:end])
        chunks.append({
            **paper,
            "chunk_text": chunk_text,
            "chunk_index": len(chunks),
            "strategy": "fixed"
        })
        start += chunk_size - overlap
    return chunks

def chunk_semantic(paper):
    """Strategy 2: Split on sentence boundaries"""
    sentences = nltk.sent_tokenize(paper["abstract"])
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))
        if current_tokens + sentence_tokens > 200 and current_chunk:
            chunks.append({
                **paper,
                "chunk_text": " ".join(current_chunk),
                "chunk_index": len(chunks),
                "strategy": "semantic"
            })
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append({
            **paper,
            "chunk_text": " ".join(current_chunk),
            "chunk_index": len(chunks),
            "strategy": "semantic"
        })
    return chunks

def chunk_sentence_window(paper, window=2):
    """Strategy 3: Embed sentences, return surrounding window"""
    sentences = nltk.sent_tokenize(paper["abstract"])
    chunks = []
    for i, sentence in enumerate(sentences):
        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        window_text = " ".join(sentences[start:end])
        chunks.append({
            **paper,
            "chunk_text": window_text,
            "chunk_index": i,
            "strategy": "sentence_window"
        })
    return chunks

# ─── EMBED + UPSERT ───────────────────────────────────────────────────

def embed_and_upsert(chunks, namespace, batch_size=100):
    total = len(chunks)
    uploaded = 0

    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["chunk_text"] for c in batch]

        try:
            response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
            embeddings = [item.embedding for item in response.data]
        except Exception as e:
            print(f"  Embedding error: {e}")
            time.sleep(10)
            continue

        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append({
                "id": f"{chunk['pmid']}_{chunk['chunk_index']}_{namespace}",
                "values": embedding,
                "metadata": {
                    "pmid": chunk["pmid"],
                    "title": chunk["title"],
                    "authors": ", ".join(chunk["authors"]),
                    "year": chunk["year"] or "Unknown",
                    "journal": chunk["journal"],
                    "disease_area": chunk["disease_area"],
                    "chunk_text": chunk["chunk_text"][:500]
                }
            })

        index.upsert(vectors=vectors, namespace=namespace)
        uploaded += len(batch)
        print(f"  [{namespace}] Uploaded {uploaded}/{total}")
        time.sleep(0.3)

# ─── EVALUATE ─────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    {"q": "What are first-line treatments for type 2 diabetes?", "disease": "diabetes"},
    {"q": "How does metformin work in diabetes?", "disease": "diabetes"},
    {"q": "What are the side effects of insulin therapy?", "disease": "diabetes"},
    {"q": "What is HbA1c and why does it matter?", "disease": "diabetes"},
    {"q": "What causes treatment-resistant depression?", "disease": "depression"},
    {"q": "How effective are SSRIs for depression?", "disease": "depression"},
    {"q": "What is cognitive behavioral therapy for depression?", "disease": "depression"},
    {"q": "What are symptoms of major depressive disorder?", "disease": "depression"},
    {"q": "How does exercise reduce cardiovascular disease risk?", "disease": "cardiovascular"},
    {"q": "What role does cholesterol play in heart disease?", "disease": "cardiovascular"},
    {"q": "What medications prevent heart attacks?", "disease": "cardiovascular"},
    {"q": "How does hypertension affect the heart?", "disease": "cardiovascular"},
]

def embed_query(query):
    response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def evaluate_namespace(namespace):
    correct = 0
    total = len(TEST_QUESTIONS)
    scores = []

    for item in TEST_QUESTIONS:
        query_vec = embed_query(item["q"])
        results = index.query(
            vector=query_vec,
            top_k=5,
            namespace=namespace,
            include_metadata=True
        )
        matches = results["matches"]

        # Check: did any top-5 result match the right disease area?
        retrieved_diseases = [m["metadata"]["disease_area"] for m in matches]
        if item["disease"] in retrieved_diseases:
            correct += 1

        # Record top similarity score
        if matches:
            scores.append(matches[0]["score"])

    precision = correct / total
    avg_score = sum(scores) / len(scores) if scores else 0
    return precision, avg_score

# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    with open("data/papers.json") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers\n")

    # Build chunks for each strategy
    print("Building chunks...")
    fixed_chunks = []
    semantic_chunks = []
    window_chunks = []

    for paper in papers:
        fixed_chunks.extend(chunk_fixed(paper))
        semantic_chunks.extend(chunk_semantic(paper))
        window_chunks.extend(chunk_sentence_window(paper))

    print(f"Fixed chunks:          {len(fixed_chunks)}")
    print(f"Semantic chunks:       {len(semantic_chunks)}")
    print(f"Sentence window chunks:{len(window_chunks)}")

    # Index each strategy into its own namespace
    print("\nIndexing fixed chunks...")
    embed_and_upsert(fixed_chunks, namespace="fixed")

    print("\nIndexing semantic chunks...")
    embed_and_upsert(semantic_chunks, namespace="semantic")

    print("\nIndexing sentence_window chunks...")
    embed_and_upsert(window_chunks, namespace="sentence_window")

    # Evaluate all 3
    print("\nEvaluating all 3 strategies...")
    results = {}
    for ns in ["fixed", "semantic", "sentence_window"]:
        precision, avg_score = evaluate_namespace(ns)
        results[ns] = {"precision": precision, "avg_score": avg_score}
        print(f"  {ns}: precision={precision:.2f}, avg_similarity={avg_score:.3f}")

    # Print comparison table
    print("\n" + "="*55)
    print(f"{'Strategy':<20} {'Precision':>10} {'Avg Score':>12}")
    print("="*55)
    for ns, r in results.items():
        print(f"{ns:<20} {r['precision']:>10.2f} {r['avg_score']:>12.3f}")
    print("="*55)

    # Save results
    with open("data/chunking_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to data/chunking_results.json")

if __name__ == "__main__":
    main()