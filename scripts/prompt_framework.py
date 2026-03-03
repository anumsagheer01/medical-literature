import os
import json
from openai import OpenAI
from pinecone import Pinecone
import cohere
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

EMBEDDING_MODEL = "text-embedding-3-small"

def load_prompt(filename):
    with open(f"prompts/{filename}") as f:
        return f.read()

def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def retrieve_and_rerank(query, top_k=20, top_n=5):
    # Pinecone retrieval
    query_vec = embed_query(query)
    matches = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace="sentence_window",
        include_metadata=True
    )["matches"]

    # Cohere rerank
    docs = [m["metadata"]["chunk_text"] for m in matches]
    reranked = co.rerank(
        query=query,
        documents=docs,
        top_n=top_n,
        model="rerank-english-v3.0"
    )

    # Return reranked matches
    return [matches[r.index] for r in reranked.results]

def build_context(matches):
    parts = []
    for i, match in enumerate(matches):
        meta = match["metadata"]
        parts.append(
            f"[{i+1}] Title: {meta['title']}\n"
            f"Authors: {meta['authors']}\n"
            f"Journal: {meta['journal']} ({meta['year']})\n"
            f"PMID: {meta['pmid']}\n"
            f"Text: {meta['chunk_text']}\n"
        )
    return "\n---\n".join(parts)

def ask_v1(question, context):
    system = load_prompt("system_v1.txt")
    user = load_prompt("user_template.txt").format(
        question=question,
        context=context
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def ask_v2(question, context):
    system = load_prompt("system_v2.txt")
    user = load_prompt("user_template.txt").format(
        question=question,
        context=context
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

def ask_v3(question, context):
    system = load_prompt("system_v3.txt")
    user = load_prompt("user_template.txt").format(
        question=question,
        context=context
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except:
        return raw

def run_comparison(question):
    print(f"\nQuestion: {question}")
    print("="*60)

    matches = retrieve_and_rerank(question)
    context = build_context(matches)

    print("\n--- V1 (Basic) ---")
    v1 = ask_v1(question, context)
    print(v1[:400])

    print("\n--- V2 (Guardrails) ---")
    v2 = ask_v2(question, context)
    print(v2[:400])

    print("\n--- V3 (Chain of Thought + JSON) ---")
    v3 = ask_v3(question, context)
    if isinstance(v3, dict):
        print(f"Confidence: {v3.get('confidence_score')}")
        print(f"Answer: {v3.get('answer', '')[:300]}")
        print(f"Uncertainty flags: {v3.get('uncertainty_flags')}")
    else:
        print(v3[:400])

def test_guardrail(question):
    print(f"\nOFF-TOPIC TEST: {question}")
    matches = retrieve_and_rerank("medical treatment")
    context = build_context(matches)
    v2 = ask_v2(question, context)
    print(v2[:300])

if __name__ == "__main__":
    # Test medical question across all 3 versions
    run_comparison("What are the most effective treatments for type 2 diabetes?")

    # Test guardrail — should reject this
    test_guardrail("What is the capital of France?")
    test_guardrail("Write me a poem about sunshine.")