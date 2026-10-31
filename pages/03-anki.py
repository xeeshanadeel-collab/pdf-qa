"""
PDF Toolkit: MCQ Generator + PDF Q&A + Spaced-Repetition Review (Anki-style), one page
Run:     streamlit run app.py
Install: pip install streamlit google-genai pypdf pandas
Key:     .streamlit/secrets.toml  ->  GEMINI_API_KEY = "your-key"
"""
import io
import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

DEFAULT_MODEL = "gemini-2.5-flash"  # editable in the sidebar
NOT_FOUND = "This information is not available in the provided PDF(s)."

# ======================= shared helpers =======================
def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


def llm(client, model, system, prompt, temperature=0.0, json_mode=False):
    cfg = dict(system_instruction=system, temperature=temperature)
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    resp = client.models.generate_content(
        model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
    )
    return (resp.text or "").strip()


@st.cache_data(show_spinner=False)
def extract_pdf(name: str, data: bytes):
    """Return list of (page_number, text) for a PDF."""
    reader = PdfReader(io.BytesIO(data))
    return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]


# ======================= MCQ generator =======================
MCQ_CHUNK_CHARS = 12000
MCQ_SYSTEM = """You are an expert exam writer. Create multiple-choice questions
strictly from the provided text. Rules:
- Each question has exactly 4 options (A-D) and exactly one correct answer.
- Distractors must be plausible, not obviously wrong.
- Do not use outside knowledge; do not write "according to the text".
- Vary the position of the correct answer.
Return ONLY a JSON array, no markdown fences. Each item:
{"question": str, "options": {"A": str, "B": str, "C": str, "D": str},
 "answer": "A"|"B"|"C"|"D", "explanation": str}"""


def split_text(text, size=MCQ_CHUNK_CHARS):
    chunks, cur = [], ""
    for p in text.split("\n"):
        if len(cur) + len(p) > size and cur:
            chunks.append(cur)
            cur = ""
        cur += p + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def parse_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    return json.loads(raw)


def valid_q(q):
    try:
        return q["answer"] in "ABCD" and set(q["options"]) == {"A", "B", "C", "D"} and q["question"]
    except Exception:
        return False


def render_mcq(client, model):
    st.subheader("📝 MCQ Generator")
    pdf = st.file_uploader("Upload a PDF", type="pdf", key="mcq_pdf")

    if pdf:
        pages = extract_pdf(pdf.name, pdf.getvalue())
        n_pages = len(pages)
        c1, c2, c3, c4 = st.columns(4)
        start = c1.number_input("From page", 1, n_pages, 1, key="mcq_from")
        end = c2.number_input("To page", 1, n_pages, n_pages, key="mcq_to")
        num_q = c3.slider("Questions", 3, 50, 10, key="mcq_n")
        difficulty = c4.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"], 1, key="mcq_d")

        if st.button("Generate MCQs", type="primary", key="mcq_go"):
            if start > end:
                st.error("'From page' must be ≤ 'To page'.")
            else:
                text = "\n".join(t for p, t in pages if start <= p <= end).strip()
                if len(text) < 200:
                    st.error("Little or no text found. Scanned PDFs need OCR first.")
                else:
                    chunks = split_text(text)[:num_q]
                    per = [num_q // len(chunks)] * len(chunks)
                    for i in range(num_q % len(chunks)):
                        per[i] += 1
                    questions, bar = [], st.progress(0.0, text="Generating...")
                    for i, (chunk, n) in enumerate(zip(chunks, per)):
                        if n == 0:
                            continue
                        try:
                            out = llm(client, model, MCQ_SYSTEM,
                                      f"Difficulty: {difficulty}\nNumber of questions: {n}\n\nTEXT:\n{chunk}",
                                      temperature=0.7, json_mode=True)
                            questions += [q for q in parse_json(out) if valid_q(q)]
                        except Exception as e:
                            st.warning(f"Chunk {i + 1} failed: {e}")
                        bar.progress((i + 1) / len(chunks), text=f"Chunk {i + 1}/{len(chunks)}")
                    bar.empty()
                    st.session_state["questions"] = questions
                    st.session_state["mcq_src"] = pdf.name

    questions = st.session_state.get("questions", [])
    if not questions:
        return
    st.success(f"{len(questions)} questions ready")
    default_deck = Path(st.session_state.get("mcq_src", "My deck")).stem
    with st.expander("🧠 Save these questions to a spaced-repetition deck"):
        deck_name = st.text_input("Deck name", value=default_deck, key="mcq_deck")
        if st.button("Save all to deck", key="mcq_save"):
            if not deck_name.strip():
                st.error("Enter a deck name.")
            else:
                added = add_cards(deck_name.strip(), questions)
                st.success(f"Added {added} new card(s) to '{deck_name.strip()}' "
                           f"({len(questions) - added} duplicate(s) skipped). "
                           "Study them in the 🧠 Review tab.")
    mode = st.radio("View", ["Study (answers shown)", "Quiz mode"], horizontal=True, key="mcq_mode")

    if mode.startswith("Study"):
        for i, q in enumerate(questions, 1):
            st.markdown(f"**Q{i}. {q['question']}**")
            for k, v in q["options"].items():
                st.write(f"{'✅' if k == q['answer'] else '▫️'} **{k}.** {v}")
            st.caption(f"💡 {q.get('explanation', '')}")
            st.divider()
    else:
        picks = {}
        for i, q in enumerate(questions):
            picks[i] = st.radio(
                f"Q{i + 1}. {q['question']}", list(q["options"]),
                format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
                index=None, key=f"pick_{i}",
            )
        if st.button("Submit quiz", key="mcq_submit"):
            score = sum(1 for i, q in enumerate(questions) if picks[i] == q["answer"])
            st.metric("Score", f"{score}/{len(questions)}")
            for i, q in enumerate(questions, 1):
                ok = picks[i - 1] == q["answer"]
                st.write(f"{'✅' if ok else '❌'} Q{i}: correct = **{q['answer']}**. {q.get('explanation', '')}")

    df = pd.DataFrame([{
        "Question": q["question"], **{f"Option {k}": v for k, v in q["options"].items()},
        "Answer": q["answer"], "Explanation": q.get("explanation", ""),
    } for q in questions])
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), "mcqs.csv", "text/csv")
    d2.download_button("⬇️ JSON", json.dumps(questions, indent=2).encode(), "mcqs.json", "application/json")


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
4. Cite sources after each claim as (File name, p. N), using the [Source | Page]
   tags in the text.
5. If different parts of the documents conflict, point out the conflict and cite both.
6. Quote figures, dates and names exactly as written. Do not guess or infer
   beyond what the text states.
7. Be concise and well organised."""

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
                st.markdown(answer)
                st.caption(f"Search mode: {meta}")
            except Exception as e:
                answer, meta = f"Error: {e}", ""
                st.error(answer)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer,
                        "meta": f"Search mode: {meta}" if meta else ""})


# ======================= Spaced repetition (Anki-style) =======================
DB_PATH = Path(__file__).resolve().parent / "srs.db"   # local SQLite file


def q(sql, params=(), one=False, write=False):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS cards(
            id INTEGER PRIMARY KEY AUTOINCREMENT, deck TEXT NOT NULL,
            question TEXT NOT NULL, options TEXT NOT NULL, answer TEXT NOT NULL,
            explanation TEXT, due TEXT NOT NULL, interval INTEGER NOT NULL DEFAULT 0,
            ease REAL NOT NULL DEFAULT 2.5, reps INTEGER NOT NULL DEFAULT 0,
            lapses INTEGER NOT NULL DEFAULT 0, UNIQUE(deck, question))""")
        cur = con.execute(sql, params)
        if write:
            con.commit()
            return cur.rowcount
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows
    finally:
        con.close()


def add_cards(deck, questions):
    added, today = 0, date.today().isoformat()
    for item in questions:
        added += q(
            "INSERT OR IGNORE INTO cards(deck, question, options, answer, explanation, due) "
            "VALUES (?,?,?,?,?,?)",
            (deck, item["question"], json.dumps(item["options"]), item["answer"],
             item.get("explanation", ""), today), write=True)
    return added


def next_state(c, rating):
    """SM-2 style scheduling. rating: 1=Again, 2=Hard, 3=Good, 4=Easy."""
    reps, interval, ease, lapses = c["reps"], c["interval"], c["ease"], c["lapses"]
    today = date.today()
    if rating == 1:
        ease = max(1.3, ease - 0.2)
        if reps > 0:
            lapses += 1
            reps = 1
        interval = 0 if reps == 0 else 1
        due = today                                  # comes back in this session
    else:
        if rating == 2:
            ease = max(1.3, ease - 0.15)
            interval = 1 if reps == 0 else max(interval + 1, round(interval * 1.2))
        elif rating == 3:
            interval = (1 if reps == 0 else 3 if reps == 1
                        else max(interval + 1, round(interval * ease)))
        else:
            ease += 0.15
            interval = 4 if reps == 0 else max(interval + 2, round(interval * ease * 1.3))
        reps += 1
        due = today + timedelta(days=interval)
    return interval, round(ease, 2), reps, lapses, due.isoformat()


def rate_card(card_id, rating):
    c = q("SELECT * FROM cards WHERE id=?", (card_id,), one=True)
    interval, ease, reps, lapses, due = next_state(c, rating)
    q("UPDATE cards SET interval=?, ease=?, reps=?, lapses=?, due=? WHERE id=?",
      (interval, ease, reps, lapses, due, card_id), write=True)


def build_queue(deck, new_limit):
    today = date.today().isoformat()
    due = q("SELECT id FROM cards WHERE deck=? AND reps>0 AND due<=? ORDER BY due, id",
            (deck, today))
    new = q("SELECT id FROM cards WHERE deck=? AND reps=0 ORDER BY id LIMIT ?",
            (deck, int(new_limit)))
    return [r["id"] for r in due] + [r["id"] for r in new]


def interval_label(c, rating):
    interval = next_state(c, rating)[0]
    if rating == 1:
        return "now"
    return f"{interval}d"


def render_srs():
    st.subheader("🧠 Spaced-repetition review")
    decks = [r["deck"] for r in q("SELECT DISTINCT deck FROM cards ORDER BY deck")]
    if not decks:
        st.info("No cards yet. Generate MCQs in the 📝 tab and save them to a deck.")
        return

    deck = st.selectbox("Deck", decks, key="srs_deck_sel")
    today = date.today().isoformat()
    total = q("SELECT COUNT(*) n FROM cards WHERE deck=?", (deck,), one=True)["n"]
    due_n = q("SELECT COUNT(*) n FROM cards WHERE deck=? AND reps>0 AND due<=?",
              (deck, today), one=True)["n"]
    new_n = q("SELECT COUNT(*) n FROM cards WHERE deck=? AND reps=0", (deck,), one=True)["n"]
    mature = q("SELECT COUNT(*) n FROM cards WHERE deck=? AND interval>=21",
               (deck,), one=True)["n"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Due today", due_n)
    m2.metric("New", new_n)
    m3.metric("Mature (21d+)", mature)
    m4.metric("Total", total)

    c1, c2 = st.columns([1, 3])
    new_limit = c1.number_input("New cards / session", 0, 200, 20, key="srs_newlim")
    if c2.button("▶ Start / restart session", type="primary", key="srs_start"):
        st.session_state.update(srs_queue=build_queue(deck, new_limit), srs_active=deck,
                                srs_revealed=False, srs_n=0)
        st.rerun()

    queue = st.session_state.get("srs_queue")
    if st.session_state.get("srs_active") != deck or queue is None:
        st.caption("Press **Start / restart session** to begin.")
    elif not queue:
        st.success("🎉 All done for now. Come back when more cards are due.")
    else:
        cid = queue[0]
        c = q("SELECT * FROM cards WHERE id=?", (cid,), one=True)
        opts = json.loads(c["options"])
        revealed = st.session_state.get("srs_revealed", False)
        st.divider()
        st.caption(f"{len(queue)} card(s) left in this session")
        st.markdown(f"### {c['question']}")
        pick = st.radio(
            "Your answer", list(opts), index=None, disabled=revealed,
            format_func=lambda k: f"{k}. {opts[k]}",
            key=f"srs_pick_{cid}_{st.session_state.get('srs_n', 0)}",
            label_visibility="collapsed",
        )
        if not revealed:
            if st.button("Show answer", type="primary", key="srs_show"):
                st.session_state["srs_revealed"] = True
                st.rerun()
        else:
            ans = c["answer"]
            if pick == ans:
                st.success(f"✅ Correct: **{ans}. {opts[ans]}**")
            elif pick is None:
                st.info(f"Answer: **{ans}. {opts[ans]}**")
            else:
                st.error(f"❌ Correct answer: **{ans}. {opts[ans]}**")
            if c["explanation"]:
                st.caption(f"💡 {c['explanation']}")
            st.write("How well did you remember it?")
            cols = st.columns(4)
            for i, name in enumerate(["Again", "Hard", "Good", "Easy"], start=1):
                if cols[i - 1].button(f"{name} · {interval_label(c, i)}", key=f"srs_rate_{i}",
                                      use_container_width=True):
                    rate_card(cid, i)
                    queue.pop(0)
                    if i == 1:
                        queue.append(cid)        # see it again before finishing
                    st.session_state["srs_revealed"] = False
                    st.session_state["srs_n"] = st.session_state.get("srs_n", 0) + 1
                    st.rerun()

    with st.expander("⚙️ Manage deck"):
        rows = q("SELECT question, due, interval, ease, reps, lapses FROM cards "
                 "WHERE deck=? ORDER BY id", (deck,))
        st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True)
        confirm = st.checkbox(f"Yes, permanently delete deck '{deck}'", key="srs_confirm")
        if st.button("Delete deck", disabled=not confirm, key="srs_delete"):
            q("DELETE FROM cards WHERE deck=?", (deck,), write=True)
            st.session_state.pop("srs_queue", None)
            st.rerun()


# ======================= main page =======================
st.set_page_config(page_title="PDF Toolkit", page_icon="📄", layout="wide")
st.title("📄 PDF Toolkit")

api_key = get_api_key()
with st.sidebar:
    model = st.text_input("Gemini model", value=DEFAULT_MODEL)

client = genai.Client(api_key=api_key) if api_key else None
NO_KEY = "GEMINI_API_KEY not found. Add it to .streamlit/secrets.toml."
tab_mcq, tab_qa, tab_srs = st.tabs(["📝 MCQ Generator", "📚 PDF Q&A", "🧠 Review"])
with tab_mcq:
    if client:
        render_mcq(client, model)
    else:
        st.error(NO_KEY)
with tab_qa:
    if client:
        render_qa(client, model)
    else:
        st.error(NO_KEY)
with tab_srs:
    render_srs()
