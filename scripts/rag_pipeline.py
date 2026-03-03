import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def retrieve(query, top_k=20):
    query_embedding = embed_query(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace="abstracts",
        include_metadata=True
    )
    return results["matches"]

def build_context(matches):
    context_parts = []
    for i, match in enumerate(matches):
        meta = match["metadata"]
        context_parts.append(
            f"[{i+1}] Title: {meta['title']}\n"
            f"Authors: {meta['authors']}\n"
            f"Journal: {meta['journal']} ({meta['year']})\n"
            f"PMID: {meta['pmid']}\n"
            f"Abstract: {meta['abstract']}\n"
        )
    return "\n---\n".join(context_parts)

def answer(query, matches):
    context = build_context(matches)

    system_prompt = """You are a clinical research assistant. 
Answer the user's medical question based ONLY on the provided research papers.
Always cite papers by their number [1], [2] etc.
If the papers don't contain enough information, say so honestly.
End your answer with a 'References' section listing the papers you cited."""

    user_prompt = f"""Question: {query}

Research Papers:
{context}

Answer the question based on the papers above."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheaper than gpt-4o, good enough for testing
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def query_pipeline(question):
    print(f"\nQuestion: {question}")
    print("Retrieving from Pinecone...")
    matches = retrieve(question, top_k=20)
    print(f"Found {len(matches)} matches, top score: {matches[0]['score']:.3f}")
    print("Generating answer...\n")
    response = answer(question, matches)
    print(response)
    return response

if __name__ == "__main__":
    test_questions = [
        "What are the first-line treatments for type 2 diabetes?",
        "What medications are used for treatment-resistant depression?",
        "How does exercise affect cardiovascular disease risk?"
    ]

    for q in test_questions:
        query_pipeline(q)
        print("\n" + "="*80 + "\n")