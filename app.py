import streamlit as st
import sqlite3
import uuid
import json
from datetime import datetime
from crag_code import app as crag_app  


# Basic page setup

st.set_page_config(
    page_title="CRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "chat_history.db"


# DB helpers — keep it simple, one connection per operation is fine here


def init_db():
    """Create tables on first run. Also handles migrating older DBs that
    don't have the sources column yet."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            verdict         TEXT,
            reason          TEXT,
            web_query       TEXT,
            sources         TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    # safe migration — silently ignore if the column already exists
    try:
        conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()


def create_conversation(title: str) -> str:
    """Start a new conversation and return its ID."""
    cid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (cid, title, now, now))
    conn.commit()
    conn.close()
    return cid


def update_conversation_title(cid: str, title: str):
    """Rename a conversation — we do this after the first message so the
    title reflects what the chat is actually about."""
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, now, cid))
    conn.commit()
    conn.close()


def save_message(cid: str, role: str, content: str,
                 verdict="", reason="", web_query="", sources=""):
    """Persist a single message and bump the conversation's updated_at."""
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages "
        "(conversation_id, role, content, verdict, reason, web_query, sources, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (cid, role, content, verdict, reason, web_query, sources, now)
    )
    conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, cid))
    conn.commit()
    conn.close()


def load_conversations() -> list:
    """Fetch all conversations sorted newest-first for the sidebar."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return rows


def load_messages(cid: str) -> list:
    """Pull all messages for a conversation in chronological order."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, verdict, reason, web_query, sources "
        "FROM messages WHERE conversation_id=? ORDER BY id ASC", (cid,)
    ).fetchall()
    conn.close()
    return rows


def delete_conversation(cid: str):
    """Wipe a conversation and all its messages from the DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
    conn.commit()
    conn.close()


def row_to_message(row) -> dict:
    """Convert a DB row tuple into a clean message dict the UI can use."""
    return {
        "role":      row[0],
        "content":   row[1],
        "verdict":   row[2] or "",
        "reason":    row[3] or "",
        "web_query": row[4] or "",
        "sources":   json.loads(row[5]) if row[5] else [],
    }


# CRAG pipeline runner — invokes the graph and packages up the sources


def run_crag(question: str) -> dict:
    """Run the full CRAG pipeline and attach a sources list to the result
    so the UI knows what to show in the sources panel."""
    result = crag_app.invoke({
        "question": question, "docs": [], "good_docs": [], "verdict": "",
        "reason": "", "strips": [], "kept_strips": [], "refined_context": "",
        "web_query": "", "web_docs": [], "answer": "",
    })

    verdict   = result.get("verdict", "")
    good_docs = result.get("good_docs", [])
    web_docs  = result.get("web_docs", [])

    sources = []

    # local document sources
    if verdict in ("CORRECT", "AMBIGUOUS"):
        for i, d in enumerate(good_docs):
            meta   = d.metadata or {}
            source = meta.get("source", "")
            page   = meta.get("page", "")
            label  = f"📄 {source.split('/')[-1]}" if source else f"📄 Document chunk {i+1}"
            if page != "":
                label += f"  (page {int(page)+1})"
            sources.append({"type": "file", "label": label, "content": d.page_content[:400]})

    # web sources
    if verdict in ("INCORRECT", "AMBIGUOUS"):
        for d in web_docs:
            meta  = d.metadata or {}
            url   = meta.get("url", "")
            title = meta.get("title", url)
            sources.append({"type": "web", "label": f"🌐 {title}", "url": url, "content": d.page_content[:400]})

    result["sources"] = sources
    return result


# UI rendering helpers


VERDICT_ICONS = {"CORRECT": "✅", "INCORRECT": "❌", "AMBIGUOUS": "⚠️"}


def render_assistant_message(msg: dict, msg_index: int):
    """Render the assistant bubble: answer text first, then an optional
    collapsible sources panel below it."""
    st.write(msg["content"])

    sources = msg.get("sources", [])
    if not sources:
        return

    # small left-aligned toggle button for the sources panel
    btn_col, _ = st.columns([1, 5])
    is_open    = st.session_state.show_sources.get(msg_index, False)
    with btn_col:
        if st.button("▲ Sources" if is_open else "▼ Sources", key=f"src_btn_{msg_index}"):
            st.session_state.show_sources[msg_index] = not is_open
            st.rerun()

    if not is_open:
        return

    # sources panel content
    with st.container(border=True):
        verdict   = msg.get("verdict", "")
        web_query = msg.get("web_query", "")

        if verdict:
            icon = VERDICT_ICONS.get(verdict, "❓")
            st.caption(f"{icon} **{verdict}** — {msg.get('reason', '')}")
        if web_query:
            st.caption(f"🔍 Web query: `{web_query}`")

        st.divider()

        file_sources = [s for s in sources if s["type"] == "file"]
        web_sources  = [s for s in sources if s["type"] == "web"]

        if file_sources:
            st.markdown("**📄 From your documents**")
            for s in file_sources:
                with st.expander(s["label"]):
                    st.caption(s["content"] + "…")

        if web_sources:
            st.markdown("**🌐 From web search**")
            for s in web_sources:
                with st.expander(s["label"]):
                    if s.get("url"):
                        st.markdown(f"[{s['url']}]({s['url']})")
                    st.caption(s["content"] + "…")


# Session state — initialize defaults on first load

init_db()

st.session_state.setdefault("conversation_id", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("show_sources", {})


# Sidebar — new chat button + conversation history list

with st.sidebar:
    st.title(" Corrective-RAG")
    st.divider()

    if st.button(" New Chat", use_container_width=True):
        st.session_state.update(conversation_id=None, messages=[], show_sources={})
        st.rerun()

    st.subheader("Conversations")
    conversations = load_conversations()

    if not conversations:
        st.caption("No conversations yet.")

    for cid, title, _ in conversations:
        is_active = (cid == st.session_state.conversation_id)
        label     = f"{'▶ ' if is_active else ''}{title[:40]}{'…' if len(title) > 40 else ''}"

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"conv_{cid}", use_container_width=True):
                st.session_state.update(
                    conversation_id=cid,
                    show_sources={},
                    messages=[row_to_message(r) for r in load_messages(cid)],
                )
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{cid}"):
                delete_conversation(cid)
                if st.session_state.conversation_id == cid:
                    st.session_state.update(conversation_id=None, messages=[], show_sources={})
                st.rerun()

    st.divider()
    st.caption("Powered by Groq · ChromaDB · Tavily")


# Main chat area — welcome screen or message history


if not st.session_state.messages:
    st.title("Corrective-RAG")
    # st.write("Ask anything about your ML documents. I'll search my knowledge base and the web to give you the most accurate answer.")
    st.divider()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render_assistant_message(msg, i)


# Chat input — handle the full question → answer → save flow


user_input = st.chat_input("Ask me anything about your documents…")

if user_input and user_input.strip():
    question = user_input.strip()

    # create a new conversation on the first message
    cid = st.session_state.conversation_id
    if cid is None:
        cid = create_conversation(question[:60])
        st.session_state.conversation_id = cid

    save_message(cid, "user", question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result  = run_crag(question)
                answer  = result.get("answer", "I don't know.")
                verdict = result.get("verdict", "")
                reason  = result.get("reason", "")
                wq      = result.get("web_query", "")
                sources = result.get("sources", [])
            except Exception as e:
                answer, verdict, reason, wq, sources = f"⚠️ Error: {e}", "", "", "", []

        msg_index     = len(st.session_state.messages)
        assistant_msg = {
            "role": "assistant", "content": answer,
            "verdict": verdict, "reason": reason,
            "web_query": wq, "sources": sources,
        }
        render_assistant_message(assistant_msg, msg_index)

    save_message(cid, "assistant", answer, verdict, reason, wq, json.dumps(sources))
    st.session_state.messages.append(assistant_msg)

    # use the first question as the conversation title
    if len(st.session_state.messages) == 2:
        update_conversation_title(cid, question[:60])

    st.rerun()