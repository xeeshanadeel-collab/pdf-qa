"""Shared helpers for the multipage PDF Toolkit (keep this file next to 01-mcq.py)."""
import io
import json
import os
import random
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import errors, types
from pypdf import PdfReader

DEFAULT_MODEL = "gemini-2.5-flash"  # editable in each page's sidebar
NOT_FOUND = "This information is not available in the provided PDF(s)."
NO_KEY = "GEMINI_API_KEY not found. Add it to .streamlit/secrets.toml."


def get_client():
    return genai.Client(api_key=get_api_key()) if get_api_key() else None


def get_api_key() -> str:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY", "")


RETRY_CODES = {429, 500, 502, 503, 504}   # temporary overload / rate-limit errors
MAX_ATTEMPTS = 6


def llm(client, model, system, prompt, temperature=0.0, json_mode=False):
    cfg = dict(system_instruction=system, temperature=temperature)
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=types.GenerateContentConfig(**cfg)
            )
            return (resp.text or "").strip()
        except errors.APIError as e:
            if e.code not in RETRY_CODES or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(min(30, 2 ** (attempt + 1)) + random.random())  # 2s, 4s, 8s, 16s, 30s


@st.cache_data(show_spinner=False)
def extract_pdf(name: str, data: bytes):
    """Return list of (page_number, text) for a PDF."""
    reader = PdfReader(io.BytesIO(data))
    return [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]



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
