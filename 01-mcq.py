"""
MCQ Generator from PDF using Google Gemini
Run:  streamlit run mcq_generator.py
Install: pip install streamlit google-genai pypdf pandas
"""
import io
import json
import os
import re

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

DEFAULT_MODEL = "gemini-3-flash-preview"  # editable in the sidebar
CHUNK_CHARS = 12000  # characters of source text per API call

SYSTEM_PROMPT = """You are an expert exam writer. Create multiple-choice questions
strictly from the provided text. Rules:
- Each question has exactly 4 options (A-D) and exactly one correct answer.
- Distractors must be plausible, not obviously wrong.
- Do not use outside knowledge; do not write "according to the text".
- Vary the position of the correct answer.
Return ONLY a JSON array, no markdown fences. Each item:
{"question": str, "options": {"A": str, "B": str, "C": str, "D": str},
 "answer": "A"|"B"|"C"|"D", "explanation": str}"""


def extract_pages(file, start: int, end: int) -> str:
    reader = PdfReader(file)
    total = len(reader.pages)
    end = min(end, total)
    texts = [(reader.pages[i].extract_text() or "") for i in range(start - 1, end)]
    return "\n".join(texts).strip()


def split_chunks(text: str, size: int = CHUNK_CHARS):
    paras, chunks, cur = text.split("\n"), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur)
            cur = ""
        cur += p + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    return json.loads(raw)


def generate(client, model, text, n, difficulty):
    resp = client.models.generate_content(
        model=model,
        contents=f"Difficulty: {difficulty}\nNumber of questions: {n}\n\nTEXT:\n{text}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.7,
        ),
    )
    return parse_json(resp.text)


def valid(q):
    try:
        return (
            q["answer"] in "ABCD"
            and set(q["options"]) == {"A", "B", "C", "D"}
            and q["question"]
        )
    except Exception:
        return False


# ---------------- UI ----------------
st.set_page_config(page_title="PDF MCQ Generator", page_icon="📝", layout="wide")
st.title("📝 PDF → MCQ Generator")

def get_api_key() -> str:
    """Read the key from .streamlit/secrets.toml, falling back to env vars."""
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


api_key = get_api_key()

with st.sidebar:
    model = st.text_input("Gemini model", value=DEFAULT_MODEL)
    num_q = st.slider("Total questions", 3, 50, 10)
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Mixed"], index=1)

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    n_pages = len(PdfReader(pdf).pages)
    pdf.seek(0)
    c1, c2 = st.columns(2)
    start = c1.number_input("From page", 1, n_pages, 1)
    end = c2.number_input("To page", 1, n_pages, n_pages)

    if st.button("Generate MCQs", type="primary"):
        if not api_key:
            st.error("GEMINI_API_KEY not found. Add it to .streamlit/secrets.toml.")
            st.stop()
        if start > end:
            st.error("'From page' must be ≤ 'To page'.")
            st.stop()

        text = extract_pages(pdf, int(start), int(end))
        if len(text) < 200:
            st.error("Little or no text found. Scanned PDFs need OCR first.")
            st.stop()

        chunks = split_chunks(text)
        # Spread questions across chunks (cap the number of chunks used)
        chunks = chunks[: num_q]
        per = [num_q // len(chunks)] * len(chunks)
        for i in range(num_q % len(chunks)):
            per[i] += 1

        client = genai.Client(api_key=api_key)
        questions, bar = [], st.progress(0.0, text="Generating...")
        for i, (chunk, n) in enumerate(zip(chunks, per)):
            if n == 0:
                continue
            try:
                questions += [q for q in generate(client, model, chunk, n, difficulty) if valid(q)]
            except Exception as e:
                st.warning(f"Chunk {i + 1} failed: {e}")
            bar.progress((i + 1) / len(chunks), text=f"Chunk {i + 1}/{len(chunks)}")
        bar.empty()

        st.session_state["questions"] = questions
        st.session_state["revealed"] = False

questions = st.session_state.get("questions", [])
if questions:
    st.success(f"{len(questions)} questions ready")
    mode = st.radio("View", ["Study (answers shown)", "Quiz mode"], horizontal=True)

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
                f"Q{i + 1}. {q['question']}",
                list(q["options"]),
                format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
                index=None,
                key=f"pick_{i}",
            )
        if st.button("Submit quiz"):
            score = sum(1 for i, q in enumerate(questions) if picks[i] == q["answer"])
            st.metric("Score", f"{score}/{len(questions)}")
            for i, q in enumerate(questions, 1):
                ok = picks[i - 1] == q["answer"]
                st.write(
                    f"{'✅' if ok else '❌'} Q{i}: correct = **{q['answer']}**. "
                    f"{q.get('explanation', '')}"
                )

    # Exports
    rows = [{
        "Question": q["question"], **{f"Option {k}": v for k, v in q["options"].items()},
        "Answer": q["answer"], "Explanation": q.get("explanation", ""),
    } for q in questions]
    df = pd.DataFrame(rows)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       "mcqs.csv", "text/csv")
    d2.download_button("⬇️ JSON", json.dumps(questions, indent=2).encode(),
                       "mcqs.json", "application/json")