import streamlit as st
import os
import json
from openai import OpenAI
from pinecone import Pinecone
import cohere
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Medical Literature",
    page_icon="⚕",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Jost:wght@200;300;400&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    font-weight: 300;
    background-color: #e8e0d0;
    color: #1a1008;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 680px !important;
    padding-top: 5rem !important;
    padding-bottom: 5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5rem;
}

.topbar-left {
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #1a1008;
    font-weight: 400;
}

.topbar-right {
    font-size: 0.65rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #9a8878;
    font-weight: 300;
}

.hero { margin-bottom: 5rem; }

.hero-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9a8878;
    margin-bottom: 1.2rem;
    font-weight: 300;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.2rem;
    font-weight: 300;
    line-height: 1.08;
    color: #1a1008;
    letter-spacing: -0.01em;
}

.hero-title-italic {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4.2rem;
    font-weight: 300;
    font-style: italic;
    line-height: 1.08;
    color: #6b4c3b;
    letter-spacing: -0.01em;
}

.hero-desc {
    font-size: 1.01rem;
    color: #4a3a2a;
    line-height: 1.75;
    font-weight: 300;
    margin-top: 1.8rem;
    max-width: 480px;
}

.search-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9a8878;
    margin-bottom: 0.8rem;
    font-weight: 300;
}

.stTextInput input {
    font-family: 'Jost', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #c0b8a8 !important;
    border-radius: 0 !important;
    padding: 0.75rem 0 !important;
    color: #1a1008 !important;
    box-shadow: none !important;
    transition: border-color 0.2s !important;
    letter-spacing: 0.01em !important;
}

.stTextInput input:focus {
    border-bottom-color: #6b4c3b !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

.stTextInput input::placeholder {
    color: #b8a898 !important;
    font-weight: 300 !important;
}

.stTextInput label { display: none !important; }
.stTextInput > div { background: transparent !important; }
.stTextInput > div > div { background: transparent !important; }

.stButton > button {
    font-family: 'Jost', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    background: #1a1008 !important;
    color: #e8e0d0 !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.85rem 2.8rem !important;
    margin-top: 1.5rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover { opacity: 0.75 !important; }

.soft-divider {
    border: none;
    border-top: 1px solid #ccc4b4;
    margin: 3rem 0;
}

.answer-label {
    font-size: 0.63rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9a8878;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.conf-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

.conf-high-dot { background: #7a9e7e; }
.conf-mid-dot  { background: #c4a15a; }
.conf-low-dot  { background: #c47a5a; }

.answer-text {
    font-size: 1rem;
    line-height: 1.95;
    color: #1a1008;
    font-weight: 300;
    letter-spacing: 0.008em;
}

.flags-section { margin-top: 1.8rem; }

.flag-item {
    font-size: 0.75rem;
    color: #9a7a4a;
    padding: 0.45rem 0;
    border-bottom: 1px solid #d8d0c0;
    letter-spacing: 0.01em;
    font-weight: 300;
}

.flag-item:last-child { border-bottom: none; }

.ref-label {
    font-size: 0.63rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9a8878;
    margin-top: 2.8rem;
    margin-bottom: 1.2rem;
}

.ref-item {
    padding: 0.9rem 0;
    border-bottom: 1px solid #d8d0c0;
}

.ref-item:last-child { border-bottom: none; }

.ref-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.88rem;
    color: #9a8878;
    margin-right: 0.4rem;
}

.ref-title {
    font-size: 0.83rem;
    color: #1a1008;
    font-weight: 300;
    line-height: 1.5;
}

.ref-meta {
    font-size: 0.7rem;
    color: #9a8878;
    margin-top: 0.25rem;
    letter-spacing: 0.02em;
}

.error-msg {
    font-size: 0.8rem;
    color: #9a4a3a;
    padding: 0.8rem 0;
    border-top: 1px solid #d0c0b8;
    margin-top: 1rem;
}

.coverage-note {
    font-size: 1rem;
    color: #9a8878;
    letter-spacing: 0.08em;
    margin-top: 1rem;
    font-weight: 300;
    border-left: 2px solid #c0a888;
    padding-left: 0.8rem;
}

.wip-note {
    font-size: 1rem;
    color: #b0a090;
    letter-spacing: 0.06em;
    margin-top: 0.4rem;
    font-weight: 300;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_clients():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
    index = pc.Index("medical-literature")
    return client, co, index


client, co, index = init_clients()


def embed_query(query):
    return client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding


def retrieve_and_rerank(query):
    vec = embed_query(query)
    matches = index.query(
        vector=vec, top_k=20,
        namespace="sentence_window",
        include_metadata=True
    )["matches"]
    docs = [m["metadata"]["chunk_text"] for m in matches]
    reranked = co.rerank(
        query=query, documents=docs,
        top_n=5, model="rerank-english-v3.0"
    )
    return [matches[r.index] for r in reranked.results]


def generate_answer(question, matches):
    ctx = "\n---\n".join([
        f"[{i+1}] Title: {m['metadata']['title']}\n"
        f"Authors: {m['metadata']['authors']}\n"
        f"Journal: {m['metadata']['journal']} ({m['metadata']['year']})\n"
        f"PMID: {m['metadata']['pmid']}\n"
        f"Text: {m['metadata']['chunk_text']}"
        for i, m in enumerate(matches)
    ])
    system = open("prompts/system_v3.txt").read()
    user = open("prompts/user_template.txt").read().format(
        question=question, context=ctx
    )
    raw = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.1,
        response_format={"type": "json_object"}
    ).choices[0].message.content
    try:
        return json.loads(raw), matches
    except Exception:
        return {"answer": raw, "confidence_score": 0.5,
                "uncertainty_flags": []}, matches


# ── Page ──────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <span class="topbar-left">Medical Literature</span>
    <span class="topbar-right">PubMed · RAG · Evidence-based</span>
</div>
<div class="hero">
    <div class="hero-label">Clinical Research Search</div>
    <div class="hero-title">Ask the literature.</div>
    <div class="hero-title-italic">Get cited answers.</div>
    <div class="hero-desc">
        Ask any clinical question across diabetes, depression,
        and cardiovascular disease. Answers are drawn directly
        from peer-reviewed PubMed papers, cited, scored for
        confidence, and honest about uncertainty.
    </div>
    <div class="coverage-note">
        Current coverage: diabetes · depression · cardiovascular disease
    </div>
    <div class="wip-note">
        More disease categories in progress.
    </div>
</div>
<div class="search-label">Your question</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "q",
    value=st.session_state.get("query", ""),
    placeholder="What would you like to know?",
)

search = st.button("Search Literature")

# ── Results ───────────────────────────────────────────────────
if search and query and len(query.strip()) >= 5:
    with st.spinner(""):
        try:
            result, matches = generate_answer(
                query, retrieve_and_rerank(query)
            )
            answer = result.get("answer", "")
            conf   = float(result.get("confidence_score", 0.5))
            flags  = result.get("uncertainty_flags", [])

            if conf >= 0.8:
                dot_class  = "conf-high-dot"
                conf_label = f"High confidence · {conf:.0%}"
            elif conf >= 0.5:
                dot_class  = "conf-mid-dot"
                conf_label = f"Moderate confidence · {conf:.0%}"
            else:
                dot_class  = "conf-low-dot"
                conf_label = f"Low confidence · {conf:.0%}"

            flags_html = ""
            if flags:
                flags_html = (
                    '<div class="flags-section">'
                    + "".join(
                        f'<div class="flag-item">— {f}</div>'
                        for f in flags
                    )
                    + '</div>'
                )

            refs_html = ""
            if conf > 0.1:
                refs_html = (
                    '<div class="ref-label">References</div>'
                    + "".join([
                        f'<div class="ref-item">'
                        f'<span class="ref-num">{i+1}</span>'
                        f'<span class="ref-title">{m["metadata"]["title"]}</span>'
                        f'<div class="ref-meta">'
                        f'{m["metadata"]["authors"]} · '
                        f'{m["metadata"]["journal"]} · '
                        f'{m["metadata"]["year"]} · '
                        f'PMID {m["metadata"]["pmid"]}'
                        f'</div></div>'
                        for i, m in enumerate(matches)
                    ])
                )

            st.markdown(
                f'<hr class="soft-divider">'
                f'<div class="answer-label">'
                f'<span class="conf-dot {dot_class}"></span>'
                f'{conf_label}</div>'
                f'<div class="answer-text">{answer}</div>'
                f'{flags_html}'
                f'{refs_html}',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.markdown(
                f'<div class="error-msg">Something went wrong  {str(e)}</div>',
                unsafe_allow_html=True
            )

elif search and query and len(query.strip()) < 5:
    st.markdown(
        '<div class="error-msg">Please enter a more specific question.</div>',
        unsafe_allow_html=True
    )