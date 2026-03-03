import os
import time
import cohere
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_query(query):
    response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    return response.data[0].embedding

def retrieve_pinecone(query, top_k=20):
    query_vec = embed_query(query)
    results = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace="sentence_window",  # using our winner
        include_metadata=True
    )
    return results["matches"]

def rerank(query, matches, top_n=5):
    documents = [m["metadata"]["chunk_text"] for m in matches]
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v3.0"
    )
    # Return reranked matches in new order
    reranked = []
    for result in response.results:
        reranked.append({
            "match": matches[result.index],
            "relevance_score": result.relevance_score
        })
    return reranked

def count_tokens(text):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def compare_with_without_reranker(query):
    matches = retrieve_pinecone(query, top_k=20)

    # WITHOUT reranker — just take top 5 from Pinecone
    without_rerank = matches[:5]
    tokens_without = sum(count_tokens(m["metadata"]["chunk_text"]) for m in without_rerank)

    # WITH reranker — rerank 20, take top 5
    reranked = rerank(query, matches, top_n=5)
    tokens_with = sum(count_tokens(r["match"]["metadata"]["chunk_text"]) for r in reranked)

    reduction = ((tokens_without - tokens_with) / tokens_without) * 100

    print(f"\nQuery: {query}")
    print(f"  Tokens WITHOUT reranker (top 5 raw): {tokens_without}")
    print(f"  Tokens WITH reranker    (top 5 reranked): {tokens_with}")
    print(f"  Token reduction: {reduction:.1f}%")
    print(f"\n  Top 3 reranked results:")
    for i, r in enumerate(reranked[:3]):
        meta = r["match"]["metadata"]
        print(f"    [{i+1}] score={r['relevance_score']:.3f} | {meta['title'][:60]}...")

    return tokens_without, tokens_with, reduction

TEST_QUESTIONS = [
    "What are first-line treatments for type 2 diabetes?",
    "How do SSRIs work for depression?",
    "How does exercise reduce cardiovascular disease risk?",
    "What is the role of metformin in diabetes management?",
    "What are symptoms of major depressive disorder?",
]

def main():
    print("Comparing retrieval with and without Cohere reranker...\n")
    print("="*60)

    total_without = 0
    total_with = 0

    for q in TEST_QUESTIONS:
        without, with_, reduction = compare_with_without_reranker(q)
        total_without += without
        total_with += with_
        time.sleep(0.5)

    overall_reduction = ((total_without - total_with) / total_without) * 100

    print("\n" + "="*60)
    print(f"OVERALL RESULTS ACROSS {len(TEST_QUESTIONS)} QUERIES")
    print("="*60)
    print(f"Total tokens without reranker: {total_without}")
    print(f"Total tokens with reranker:    {total_with}")
    print(f"Overall token reduction:       {overall_reduction:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()