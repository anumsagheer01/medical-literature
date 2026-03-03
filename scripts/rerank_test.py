import os
import cohere
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

query = "What are first-line treatments for type 2 diabetes?"

vec = client.embeddings.create(
    input=[query],
    model="text-embedding-3-small"
).data[0].embedding

matches = index.query(
    vector=vec,
    top_k=20,
    namespace="sentence_window",
    include_metadata=True
)["matches"]

print("TOP 5 RAW PINECONE SCORES:")
for m in matches[:5]:
    print(f"  {m['score']:.3f} | {m['metadata']['title'][:60]}")

docs = [m["metadata"]["chunk_text"] for m in matches]
reranked = co.rerank(
    query=query,
    documents=docs,
    top_n=5,
    model="rerank-english-v3.0"
)

print("\nTOP 5 AFTER RERANKING:")
for r in reranked.results:
    print(f"  {r.relevance_score:.3f} | {matches[r.index]['metadata']['title'][:60]}")