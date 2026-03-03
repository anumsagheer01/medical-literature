import os
import json
from openai import OpenAI
from pinecone import Pinecone
import cohere
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai

load_dotenv()

# Wrap OpenAI client — this is all you need for automatic tracing
client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

EMBEDDING_MODEL = "text-embedding-3-small"

@traceable(name="embed_query")
def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

@traceable(name="pinecone_retrieve")
def retrieve_pinecone(query_vec, top_k=20):
    return index.query(
        vector=query_vec,
        top_k=top_k,
        namespace="sentence_window",
        include_metadata=True
    )["matches"]

@traceable(name="cohere_rerank")
def rerank(query, matches, top_n=5):
    docs = [m["metadata"]["chunk_text"] for m in matches]
    reranked = co.rerank(
        query=query,
        documents=docs,
        top_n=top_n,
        model="rerank-english-v3.0"
    )
    return [matches[r.index] for r in reranked.results]

@traceable(name="generate_answer")
def generate_answer(question, matches):
    context_parts = []
    for i, match in enumerate(matches):
        meta = match["metadata"]
        context_parts.append(
            f"[{i+1}] Title: {meta['title']}\n"
            f"Authors: {meta['authors']}\n"
            f"Journal: {meta['journal']} ({meta['year']})\n"
            f"Text: {meta['chunk_text']}\n"
        )
    context = "\n---\n".join(context_parts)

    system = open("prompts/system_v3.txt").read()
    user = open("prompts/user_template.txt").read().format(
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
        parsed = json.loads(raw)
        return parsed
    except:
        return {"answer": raw}

@traceable(name="full_rag_pipeline")
def run_pipeline(question):
    query_vec = embed_query(question)
    matches = retrieve_pinecone(query_vec)
    reranked = rerank(question, matches)
    result = generate_answer(question, reranked)
    return result

# Test questions covering all 3 failure patterns to find
TEST_QUESTIONS = [
    # Normal questions — should work well
    "What are the first-line treatments for type 2 diabetes?",
    "How do SSRIs work for depression?",
    "How does exercise reduce cardiovascular disease risk?",
    # Edge cases — to find failure patterns
    "What is the latest research on tirzepatide for obesity?",
    "How does transcranial magnetic stimulation treat depression?",
    "What are the genetic risk factors for cardiovascular disease?",
    "Compare metformin and SGLT2 inhibitors for diabetes treatment",
    "What happens to the brain during a depressive episode?",
    "How do omega-3 fatty acids affect heart disease risk?",
    # Off-topic — guardrail test
    "What is the weather today?",
]

def main():
    print("Running traced queries — check LangSmith dashboard after\n")
    print("Dashboard: https://smith.langchain.com\n")
    print("="*60)

    for q in TEST_QUESTIONS:
        print(f"\nQ: {q}")
        try:
            result = run_pipeline(q)
            if isinstance(result, dict):
                answer = result.get("answer", "")
                confidence = result.get("confidence_score", "N/A")
                flags = result.get("uncertainty_flags", [])
                print(f"Confidence: {confidence}")
                print(f"Answer preview: {str(answer)[:150]}...")
                if flags:
                    print(f"Flags: {flags}")
            else:
                print(f"Result: {str(result)[:150]}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "="*60)
    print("Done. Now go to https://smith.langchain.com")
    print("Click your 'medical-literature' project")
    print("You should see all 10 traced queries with full details")
    print("="*60)

if __name__ == "__main__":
    main()