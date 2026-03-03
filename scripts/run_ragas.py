import os
import json
import time
import numpy as np
from openai import OpenAI
from pinecone import Pinecone
import cohere
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("medical-literature")

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_query(query):
    response = client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def retrieve_and_rerank(query, top_k=20, top_n=5):
    query_vec = embed_query(query)
    matches = index.query(
        vector=query_vec,
        top_k=top_k,
        namespace="sentence_window",
        include_metadata=True
    )["matches"]

    docs = [m["metadata"]["chunk_text"] for m in matches]
    reranked = co.rerank(
        query=query,
        documents=docs,
        top_n=top_n,
        model="rerank-english-v3.0"
    )
    return [matches[r.index] for r in reranked.results]

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
        return parsed.get("answer", raw)
    except:
        return raw

def get_score(val):
    if isinstance(val, list):
        return float(np.nanmean([v for v in val if v is not None]))
    return float(val)

def main():
    # Configure RAGAS
    langchain_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    langchain_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    metrics = [
            Faithfulness(llm=langchain_llm),
            AnswerRelevancy(llm=langchain_llm, embeddings=langchain_embeddings),
            ContextPrecision(llm=langchain_llm),
            ContextRecall(llm=langchain_llm)
        ]

    with open("eval/eval_set.json") as f:
        eval_set = json.load(f)

    # Check if answers already cached
    cache_file = "eval/answers_cache.json"
    if os.path.exists(cache_file):
        print("Found cached answers, skipping generation...")
        with open(cache_file) as f:
            cache = json.load(f)
        questions = cache["questions"]
        answers = cache["answers"]
        contexts = cache["contexts"]
        ground_truths = cache["ground_truths"]
    else:
        print(f"Generating answers for {len(eval_set)} questions...")
        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for i, item in enumerate(eval_set):
            print(f"  Processing {i+1}/{len(eval_set)}: {item['question'][:50]}...")
            try:
                matches = retrieve_and_rerank(item["question"])
                answer = generate_answer(item["question"], matches)
                context_texts = [m["metadata"]["chunk_text"] for m in matches]

                questions.append(item["question"])
                answers.append(answer)
                contexts.append(context_texts)
                ground_truths.append(item["ground_truth"])

            except Exception as e:
                print(f"  Error on question {i+1}: {e}")
                continue

            time.sleep(0.5)

        # Save cache so we never regenerate
        with open(cache_file, "w") as f:
            json.dump({
                "questions": questions,
                "answers": answers,
                "contexts": contexts,
                "ground_truths": ground_truths
            }, f, indent=2)
        print(f"Answers cached to {cache_file}")

    print(f"\nBuilding dataset with {len(questions)} questions...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("Running RAGAS evaluation...")
    results = evaluate(dataset, metrics=metrics)

    f  = get_score(results["faithfulness"])
    ar = get_score(results["answer_relevancy"])
    cp = get_score(results["context_precision"])
    cr = get_score(results["context_recall"])

    print("\n" + "="*55)
    print("RAGAS EVALUATION RESULTS")
    print("="*55)
    print(f"Faithfulness:      {f:.3f}")
    print(f"Answer Relevancy:  {ar:.3f}")
    print(f"Context Precision: {cp:.3f}")
    print(f"Context Recall:    {cr:.3f}")
    print("="*55)

    output = {
        "num_questions": len(questions),
        "faithfulness": f,
        "answer_relevancy": ar,
        "context_precision": cp,
        "context_recall": cr
    }

    with open("eval/ragas_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved to eval/ragas_results.json")

if __name__ == "__main__":
    main()