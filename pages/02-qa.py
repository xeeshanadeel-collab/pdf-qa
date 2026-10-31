"""Page 2: PDF Q&A (answers only from the PDFs)."""
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from common import DEFAULT_MODEL, NOT_FOUND, NO_KEY, extract_pdf, get_client, llm

# ======================= PDF Q&A =======================
FULL_CONTEXT_CHARS = 400_000
QA_CHUNK_CHARS = 150_000

ANSWER_SYSTEM = f"""You answer questions using ONLY the document text provided.
Strict rules:
1. Use nothing but the provided text. Never use outside or general knowledge,
   even if you know the answer or it seems obvious.
2. If the answer is not clearly present in the text, reply exactly:
   "{NOT_FOUND}"
3. If only part of the question is answered, answer that part and explicitly
   state which part is not available in the PDF(s).
4. Do NOT put citations or page references inside the answer text. Write the
   information itself, clearly and completely. Do not start with phrases like
   "Based on the provided document". After the answer, add ONE final line in
   exactly this form (use the [File | Page N] tags of the passages you used):
   SOURCES: File name p. N, File name p. N
   Do not add a SOURCES line when you reply with the not-available message.
5. If different parts of the documents conflict, point out the conflict and cite both.
6. Quote figures, dates and names exactly as written. Do not guess or infer
   beyond what the text states.
7. Give the complete relevant information from the text (conditions, percentages,
   limits, exceptions, steps). Do not shorten it into a vague summary.
   Be well organised."""

EXTRACT_SYSTEM = """You are a meticulous search tool. Given a question and a section of
document text, copy out EVERY passage relevant to answering the question
(definitions, numbers, table rows, conditions, exceptions), verbatim, each
prefixed with its [Source | Page] tag. Do not answer the question and do not add
any outside knowledge. If nothing in this section is relevant, reply exactly: NONE"""


def build_blocks(files):
    blocks, empty = [], []
    for name, data in files:
        for page_no, text in extract_pdf(name, data):
            if text:
                blocks.append(f"[{name} | Page {page_no}]\n{text}")
            else:
                empty.append((name, page_no))
    return blocks, empty


def make_chunks(blocks):
    chunks, cur, size = [], [], 0
    for b in blocks:
        if size + len(b) > QA_CHUNK_CHARS and cur:
            chunks.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(b)
        size += len(b)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def split_sources(text):
    """Separate the answer body from the trailing 'SOURCES: ...' line."""
    m = re.search(r"\n?\s*\**SOURCES:?\**:?\s*(.+?)\s*$", text, flags=re.I | re.S)
    if not m:
        return text.strip(), ""
    return text[:m.start()].strip(), m.group(1).strip()


def standalone_question(client, model, question, history):
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = (
        "Rewrite the user's latest message as a single self-contained question, "
        "using the chat history only to resolve references (it, they, that, etc.). "
        "If it is already self-contained, return it unchanged. Return ONLY the question.\n\n"
        f"Chat history:\n{convo}\n\nLatest message: {question}"
    )
    return llm(client, model, "You rewrite questions.", prompt) or question


def answer_question(client, model, question, blocks, status):
    if sum(len(b) for b in blocks) <= FULL_CONTEXT_CHARS:
        status.update(label="Reading the full document...")
        docs = "\n\n".join(blocks)
        return llm(client, model, ANSWER_SYSTEM,
                   f"DOCUMENT TEXT:\n{docs}\n\nQUESTION: {question}"), "full document"

    chunks = make_chunks(blocks)
    status.update(label=f"Scanning all {len(chunks)} sections of the document...")

    def scan(chunk):
        out = llm(client, model, EXTRACT_SYSTEM, f"QUESTION: {question}\n\nDOCUMENT SECTION:\n{chunk}")
        return "" if out.strip().upper().startswith("NONE") else out

    with ThreadPoolExecutor(max_workers=4) as pool:
        found = [x for x in pool.map(scan, chunks) if x]
    if not found:
        return NOT_FOUND, f"scanned {len(chunks)} sections"

    status.update(label="Writing the answer...")
    passages = "\n\n".join(found)
    return llm(client, model, ANSWER_SYSTEM,
               f"DOCUMENT TEXT (relevant passages from the full document):\n{passages}"
               f"\n\nQUESTION: {question}"), f"scanned {len(chunks)} sections"


def render_qa(client, model):
    st.subheader("📚 Ask your PDFs")
    st.caption("Answers come only from the uploaded PDFs. If it isn't in them, the app says so.")
    uploads = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True, key="qa_pdf")
    history = st.session_state.setdefault("history", [])

    if st.button("Clear chat", key="qa_clear"):
        history.clear()
        st.rerun()

    if not uploads:
        st.info("Upload one or more PDFs to begin.")
        return

    files = [(f.name, f.getvalue()) for f in uploads]
    blocks, empty = build_blocks(files)
    if not blocks:
        st.error("No extractable text found. These look like scanned PDFs and need OCR first.")
        return

    st.caption(f"{len(files)} file(s) · {len(blocks)} pages with text · "
               f"~{sum(len(b) for b in blocks) // 4:,} tokens")
    if empty:
        pages = ", ".join(f"{n} p.{p}" for n, p in empty[:10])
        st.warning(f"{len(empty)} page(s) have no extractable text and cannot be searched: "
                   f"{pages}{'...' if len(empty) > 10 else ''}")

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    if q := st.chat_input("Ask a question about the PDF(s)", key="qa_input"):
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.status("Searching...", expanded=False) as status:
                    sq = standalone_question(client, model, q, history)
                    answer, meta = answer_question(client, model, sq, blocks, status)
                    status.update(label="Done", state="complete")
                answer, sources = split_sources(answer)
                note = (f"📄 Sources: {sources}  ·  " if sources else "") + f"Search mode: {meta}"
                st.markdown(answer)
                st.caption(note)
            except Exception as e:
                answer, note = f"Error: {e}", ""
                st.error(answer)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer, "meta": note})



st.set_page_config(page_title="PDF Q&A", page_icon="📚", layout="wide")
st.title("📚 Ask your PDFs")
model = st.sidebar.text_input("Gemini model", value=DEFAULT_MODEL)
client = get_client()
if client is None:
    st.error(NO_KEY)
    st.stop()
render_qa(client, model)