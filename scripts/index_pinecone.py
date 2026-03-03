import json
import os
import time
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

INDEX_NAME = "medical-literature"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
BATCH_SIZE = 100

def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]

def create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print("Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)  # wait for index to be ready
        print("Index created.")
    else:
        print("Index already exists, skipping creation.")

def index_papers(papers, namespace):
    index = pc.Index(INDEX_NAME)
    total = len(papers)
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch = papers[i:i+BATCH_SIZE]
        texts = [p["abstract"] for p in batch]

        try:
            embeddings = get_embeddings(texts)
        except Exception as e:
            print(f"  Embedding error on batch {i//BATCH_SIZE + 1}: {e}")
            time.sleep(10)
            continue

        vectors = []
        for j, (paper, embedding) in enumerate(zip(batch, embeddings)):
            vectors.append({
                "id": f"{paper['pmid']}_{namespace}",
                "values": embedding,
                "metadata": {
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "authors": ", ".join(paper["authors"]),
                    "year": paper["year"] or "Unknown",
                    "journal": paper["journal"],
                    "disease_area": paper["disease_area"],
                    "abstract": paper["abstract"][:500]  # store first 500 chars for display
                }
            })

        index.upsert(vectors=vectors, namespace=namespace)
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{total}")
        time.sleep(0.5)  # be nice to the API

    print(f"Done indexing {total} papers into namespace: {namespace}")

def main():
    with open("data/papers.json") as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers")

    # Step 1: create index
    create_index()

    # Step 2: index into 'abstracts' namespace
    print(f"\nIndexing into 'abstracts' namespace...")
    index_papers(papers, namespace="abstracts")

    # Step 3: verify
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"\nIndex stats: {stats}")

if __name__ == "__main__":
    main()