from typing import List, TypedDict
from pydantic import BaseModel, Field, field_validator
import re
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()


# Vector store setup — load from disk if it already exists, otherwise build it
# from scratch by reading the PDFs and chunking them up.

CHROMA_PATH     = "./chroma_db"
COLLECTION_NAME = "rag_documents"

# Using mpnet here because it tends to give better retrieval quality than
# the lighter minilm models, still runs fine on CPU for this use case.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
    print("[Chroma] Loading existing database from disk...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    print(f"[Chroma] Loaded {vector_store._collection.count()} chunks.")

else:
    print("[Chroma] Building database for the first time — this may take a minute...")

    # Load all three books and combine them into one big list of pages
    docs = (
        PyPDFLoader("./documents/book1.pdf").load()
        + PyPDFLoader("./documents/book2.pdf").load()
        + PyPDFLoader("./documents/book3.pdf").load()
    )

    # Chunk size 900 with 150 overlap works well — keeps context while
    # still giving the LLM manageable pieces to score and filter
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(docs)

    # Strip out any non-UTF-8 bytes that sneak in from scanned PDFs
    for d in chunks:
        d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    print(f"[Chroma] Saved {len(chunks)} chunks to ./chroma_db")

# Top-4 by cosine similarity is usually enough — going higher adds noise
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,   # keep it focused, we're doing factual QA not creative writing
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Thresholds for classifying how good our retrieved docs are
UPPER_TH = 0.7   # "good enough to answer from"
LOWER_TH = 0.3   # "basically useless"



# Groq's structured output uses function-calling under the hood, which means
# raw PDF text with angle brackets, weird unicode, or repetitive numeric
# patterns (like table-of-contents pages) can corrupt the tool-call XML
# and cause 400 errors. These two helpers catch that before it happens.


def is_garbage_chunk(text: str) -> bool:
    """Sniff out chunks that are likely to be garbage — TOC pages, index
    sections, or anything that's mostly numbers/punctuation rather than
    actual readable content. Returns True if we should skip this chunk."""

    # repetitive dot-separated numbers are a dead giveaway for TOC/index rows
    if re.search(r'(\d+\.){10,}', text):
        return True

    # if less than 30% of the characters are actual letters, it's probably
    # a table or formula dump we don't want to feed to the LLM
    letters = len(re.findall(r'[a-zA-Z]', text))
    total   = len(text.strip())
    if total > 50 and letters / total < 0.30:
        return True

    # after stripping everything except letters and spaces, if there's
    # barely anything left then there's nothing meaningful here
    meaningful = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    if len(meaningful) < 30:
        return True

    return False


def sanitize(text: str, max_chars: int = 800) -> str:
    """Light cleanup before sending text to Groq structured output.
    Removes angle-bracket patterns that can break the function-call parser,
    collapses whitespace, and truncates to avoid token overflow."""
    text = re.sub(r"<[^>]{0,60}>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]



# Graph state — everything that flows between nodes lives here


class State(TypedDict):
    question:str
    docs:List[Document]   # raw retrieval results
    good_docs:List[Document]   # docs that passed the scoring filter
    verdict:str              # CORRECT / INCORRECT / AMBIGUOUS
    reason:str
    strips:List[str]        # all sentences from the context
    kept_strips:List[str]        # sentences the LLM filter kept
    refined_context:str              # final cleaned context for generation
    web_query:str
    web_docs:List[Document]
    answer:str



# Node 1 — pull the top-k docs from Chroma for the user's question


def retrieve_node(state: State) -> State:
    """Simple retrieval step — just hit the vector store and return whatever
    comes back. Scoring and filtering happen in the next node."""
    q = state["question"]
    return {"docs": retriever.invoke(q)}



# Node 2 — score each retrieved doc and decide if our local KB is useful


class DocEvalScore(BaseModel):
    score: float = Field(
        description="Relevance score between 0.0 (irrelevant) and 1.0 (perfectly answers the question)."
    )
    reason: str = Field(
        description="One-sentence explanation of the score."
    )

doc_eval_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict retrieval evaluator for RAG.\n"
     "You will be given ONE retrieved chunk and a question.\n"
     "Return a relevance score in [0.0, 1.0].\n"
     "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
     "- 0.0: chunk is irrelevant\n"
     "Be conservative with high scores.\n"
     "Also return a short reason."),
    ("human", "Question: {question}\n\nChunk:\n{chunk}"),
])

doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state: State) -> State:
    """Score every retrieved chunk individually, then roll up into one of
    three verdicts: CORRECT (we have what we need), INCORRECT (our KB is
    useless here), or AMBIGUOUS (partial match, probably want web too)."""
    q      = state["question"]
    scores: List[float]    = []
    good:   List[Document] = []

    for d in state["docs"]:
        if is_garbage_chunk(d.page_content):
            print("[eval_each_doc] skipping garbage chunk")
            scores.append(0.0)
            continue

        try:
            clean_chunk = sanitize(d.page_content, max_chars=800)
            out: DocEvalScore = doc_eval_chain.invoke({
                "question": q,
                "chunk":    clean_chunk,
            })
            scores.append(out.score)
            if out.score > LOWER_TH:
                good.append(d)

        except Exception as e:
            print(f"[eval_each_doc] skipping chunk due to error: {e}")
            scores.append(0.0)

    # at least one chunk is strong enough to answer from — we're good
    if any(s > UPPER_TH for s in scores):
        return {
            "good_docs": good,
            "verdict":   "CORRECT",
            "reason":    f"At least one retrieved chunk scored > {UPPER_TH}.",
        }

    # everything was below the floor — local KB has nothing useful
    if scores and all(s < LOWER_TH for s in scores):
        return {
            "good_docs": [],
            "verdict":   "INCORRECT",
            "reason":    f"All retrieved chunks scored < {LOWER_TH}.",
        }

    # somewhere in the middle — keep partial matches and augment with web
    return {
        "good_docs": good,
        "verdict":   "AMBIGUOUS",
        "reason":    f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
    }



# Sentence splitting helper — used inside refine() to work at a finer grain


def decompose_to_sentences(text: str) -> List[str]:
    """Split a block of text into individual sentences. We ignore anything
    shorter than 20 chars since those are usually headers or stray fragments."""
    text      = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]



# Node — sentence-level relevance filter (runs inside refine)


class KeepOrDrop(BaseModel):
    keep: bool = Field(
        description="Must be a boolean true or false — NOT a string. "
                    "true if the sentence directly helps answer the question, false otherwise."
    )

    @field_validator("keep", mode="before")
    @classmethod
    def coerce_to_bool(cls, v):
        """Groq occasionally returns the string 'true' instead of a real bool —
        this validator handles that gracefully so we don't crash downstream."""
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return v

filter_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance filter.\n"
     "Return keep=true only if the sentence directly helps answer the question.\n"
     "keep must be a boolean (true or false), NOT a string.\n"
     "Use ONLY the sentence."),
    ("human", "Question: {question}\n\nSentence:\n{sentence}"),
])

filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)



# Node 3 — refine the context down to only the sentences that actually matter
# Source selection depends on what the evaluator decided:
#   CORRECT   - internal docs only
#   INCORRECT - web results only
#   AMBIGUOUS - blend both


def refine(state: State) -> State:
    """Sentence-level context refinement. We decompose all the candidate docs
    into individual sentences and ask the LLM to keep only the ones that
    genuinely help answer the question. Cuts down on noise in the final prompt."""
    q = state["question"]

    if state.get("verdict") == "CORRECT":
        docs_to_use = state["good_docs"]
    elif state.get("verdict") == "INCORRECT":
        docs_to_use = state["web_docs"]
    else:
        docs_to_use = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs_to_use).strip()
    strips  = decompose_to_sentences(context)

    kept: List[str] = []
    for s in strips:
        if is_garbage_chunk(s):
            continue

        try:
            clean_s = sanitize(s, max_chars=400)
            out: KeepOrDrop = filter_chain.invoke({
                "question": q,
                "sentence": clean_s,
            })
            if out.keep:
                kept.append(s)
        except Exception as e:
            print(f"[refine] skipping sentence due to error: {e}")
            continue

    return {
        "strips":          strips,
        "kept_strips":     kept,
        "refined_context": "\n".join(kept).strip(),
    }



# Node 4 — rewrite the question into a better web search query


class WebQuery(BaseModel):
    query: str = Field(
        description="Short web search query (6-14 words) derived from the user question."
    )

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a web search query composed of keywords.\n"
     "Rules:\n"
     "- Keep it short (6-14 words).\n"
     "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
     "- Do NOT answer the question.\n"
     "- Return JSON with a single key: query"),
    ("human", "Question: {question}"),
])

rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state: State) -> State:
    """Turn the user's natural language question into a tight keyword query
    for Tavily. Helps a lot for technical topics where phrasing matters."""
    out: WebQuery = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}



# Node 5 — hit the web and turn results into Documents


tavily = TavilySearchResults(max_results=5)


def web_search_node(state: State) -> State:
    """Run Tavily search with the rewritten query and package the results as
    LangChain Documents so the rest of the pipeline can treat them the same
    as the locally retrieved chunks."""
    q       = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})

    web_docs: List[Document] = []
    for r in results or []:
        title   = r.get("title", "")
        url     = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        # bundle title + URL into the text so the LLM has provenance info
        text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

    return {"web_docs": web_docs}



# Node 6 — final answer generation


answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful ML tutor. Answer ONLY using the provided context.\n"
     "If the context is empty or insufficient, say: 'I don't know.'"),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def generate(state: State) -> State:
    """Generate the final answer from whatever context survived the refine step.
    Keeping the prompt short and grounded — no hallucination allowed."""
    out = (answer_prompt | llm | StrOutputParser()).invoke({
        "question": state["question"],
        "context":  state["refined_context"],
    })
    return {"answer": out}



# Router — decides whether we need web search or can go straight to refine


def route_after_eval(state: State) -> str:
    """If we got good docs from the local KB, skip web search entirely.
    Otherwise go through query rewrite → web search first."""
    if state["verdict"] == "CORRECT":
        return "refine"
    return "rewrite_query"



# Wire everything together into a LangGraph


g = StateGraph(State)

g.add_node("retrieve",      retrieve_node)
g.add_node("eval_each_doc", eval_each_doc_node)
g.add_node("rewrite_query", rewrite_query_node)
g.add_node("web_search",    web_search_node)
g.add_node("refine",        refine)
g.add_node("generate",      generate)

g.add_edge(START,          "retrieve")
g.add_edge("retrieve",     "eval_each_doc")

# branch here depending on how good our local docs are
g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine":        "refine",
        "rewrite_query": "rewrite_query",
    },
)

# non-correct path goes through web before refining
g.add_edge("rewrite_query", "web_search")
g.add_edge("web_search",    "refine")

# both paths converge at refine → generate
g.add_edge("refine",   "generate")
g.add_edge("generate", END)

app = g.compile()



# Quick smoke test — only runs when you execute this file directly


if __name__ == "__main__":
    res = app.invoke(
        {
            "question":"Types of activation function",
            "docs":[],
            "good_docs":[],
            "verdict":"",
            "reason":"",
            "strips":[],
            "kept_strips":[],
            "refined_context": "",
            "web_query":"",
            "web_docs":[],
            "answer":"",
        }
    )

    print("VERDICT:  ", res["verdict"])
    print("REASON:   ", res["reason"])
    print("WEB_QUERY:", res["web_query"])
    print("\nOUTPUT:\n", res["answer"])