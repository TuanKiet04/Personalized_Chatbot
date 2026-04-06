import streamlit as st
import requests
import json
import psycopg2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DB", "optimize"),
    "user":     os.getenv("PG_USER", "kietcorn"),
    "password": os.getenv("PG_PASS", "kiietqo9204"),
}

OLLAMA_BASE  = os.getenv("OLLAMA_URL", "http://10.4.21.3:11435")
OLLAMA_URL   = f"{OLLAMA_BASE}/api/chat"
EMBED_URL    = f"{OLLAMA_BASE}/api/embeddings"
OLLAMA_MODEL = "qwen3.5:4b"
EMBED_MODEL  = "nomic-embed-text:latest"


# ---------- helpers ----------

@st.cache_data
def load_personas():
    with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_query(text):
    res = requests.post(f"{OLLAMA_URL}/embeddings", json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    res.raise_for_status()
    return res.json()["embedding"]


def retrieve(query, top_k=5):
    vec = embed_query(query)
    vec_str = "[" + ",".join(map(str, vec)) + "]"
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT text, 1 - (embedding <=> '{vec_str}'::vector) AS score
            FROM n8n_vectors
            WHERE text LIKE 'Content:%%' OR text LIKE 'Title:%%'
            ORDER BY embedding <=> '{vec_str}'::vector
            LIMIT {top_k}
        """)
        rows = cur.fetchall()
    conn.close()
    return [{"text": r[0], "score": round(r[1], 4)} for r in rows]


def call_llm_stream(system_prompt, user_query, context_chunks):
    context = "\n\n".join([c["text"] for c in context_chunks])
    full_prompt = f"""Dưới đây là các đoạn thông tin liên quan:\n\n{context}\n\nCâu hỏi: {user_query}"""

    res = requests.post(f"{OLLAMA_URL}/chat", json={
        "model": OLLAMA_MODEL, # Lưu ý: Sửa LLM_MODEL thành OLLAMA_MODEL
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": full_prompt}
        ]
    }, stream=True)
    
    for line in res.iter_lines():
        if line:
            chunk = json.loads(line)
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]


DEFAULT_SYSTEM_PROMPT = """
VAI TRÒ: Bạn là chuyên gia phân tích tin tức cá nhân hóa. Nhiệm vụ của bạn là hỗ trợ người dùng dựa trên thông tin bài báo được cung cấp.

NGUYÊN TẮC HOẠT ĐỘNG:
1. Chỉ sử dụng thông tin từ "NGỮ CẢNH BÀI BÁO" để trả lời. Không tự bịa đặt thông tin bên ngoài.
2. Nếu thông tin không có trong ngữ cảnh, hãy lịch sự từ chối và đề nghị người dùng đọc thêm các tin liên quan.
3. Câu trả lời phải ngắn gọn, súc tích nhưng đầy đủ ý chính.

CÁ NHÂN HÓA (QUAN TRỌNG):
- Luôn điều chỉnh văn phong, tông giọng và cách đặt vấn đề dựa trên "HỒ SƠ PERSONA" ($M_s$) và "LỊCH SỬ ĐÃ ĐỌC" ($M_l$) được cung cấp.
- Nếu người dùng thuộc nhóm chuyên gia, hãy dùng thuật ngữ chuyên môn. Nếu là nhóm tin đời sống, hãy dùng văn phong gần gũi.
"""

# ---------- UI ----------

st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("RAG Chatbot Demo")

personas = load_personas()
persona_names = {p["cluster_id"]: p["persona"].get("name", f"Persona {p['cluster_id']}") for p in personas}

tab1, tab2, tab3 = st.tabs(["Baseline RAG", "RAG + Persona", "So sánh"])


# ---------- Tab 1: Baseline ----------

with tab1:
    st.subheader("Baseline RAG")
    st.caption("Chỉ retrieve + answer, không có Persona")

    with st.expander("System Prompt", expanded=True):
        st.code(DEFAULT_SYSTEM_PROMPT, language=None)

    query1 = st.text_input("Nhập câu hỏi", key="q1")

    if st.button("Gửi", key="btn1") and query1:
        with st.spinner("Đang xử lý..."):
            chunks = retrieve(query1)
            answer = call_llm(DEFAULT_SYSTEM_PROMPT, query1, chunks)

        st.markdown("**Trả lời:**")
        st.write(answer)

        with st.expander("Chunks được retrieve"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()


# ---------- Tab 2: RAG + Persona ----------

with tab2:
    st.subheader("RAG + Persona")
    st.caption("Retrieve + answer với Persona được chọn")

    selected = st.selectbox(
        "Chọn Persona",
        options=[p["cluster_id"] for p in personas],
        format_func=lambda x: persona_names[x]
    )

    persona_data = next(p for p in personas if p["cluster_id"] == selected)
    system_prompt_persona = persona_data["persona"].get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    description = persona_data["persona"].get("description", "")

    with st.expander("Thông tin Persona", expanded=True):
        st.markdown(f"**Tên:** {persona_data['persona'].get('name', '')}")
        st.markdown(f"**Mô tả:** {description}")
        st.markdown("**System Prompt:**")
        st.code(system_prompt_persona, language=None)

    query2 = st.text_input("Nhập câu hỏi", key="q2")

    if st.button("Gửi", key="btn2") and query2:
        with st.spinner("Đang xử lý..."):
            chunks = retrieve(query2)
            answer = call_llm(system_prompt_persona, query2, chunks)

        st.markdown("**Trả lời:**")
        st.write(answer)

        with st.expander("Chunks được retrieve"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()


# ---------- Tab 3: So sánh ----------

with tab3:
    st.subheader("So sánh Baseline vs Persona")

    selected_compare = st.selectbox(
        "Chọn Persona để so sánh",
        options=[p["cluster_id"] for p in personas],
        format_func=lambda x: persona_names[x],
        key="compare_persona"
    )

    persona_compare = next(p for p in personas if p["cluster_id"] == selected_compare)
    system_prompt_compare = persona_compare["persona"].get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    query3 = st.text_input("Nhập câu hỏi", key="q3")

    if st.button("So sánh", key="btn3") and query3:
        with st.spinner("Đang xử lý..."):
            chunks = retrieve(query3)
            answer_baseline = call_llm(DEFAULT_SYSTEM_PROMPT, query3, chunks)
            answer_persona  = call_llm(system_prompt_compare, query3, chunks)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Baseline**")
            with st.expander("System Prompt"):
                st.code(DEFAULT_SYSTEM_PROMPT, language=None)
            st.write(answer_baseline)

        with col2:
            st.markdown(f"**{persona_compare['persona'].get('name', 'Persona')}**")
            with st.expander("System Prompt"):
                st.code(system_prompt_compare, language=None)
            st.write(answer_persona)

        with st.expander("Chunks được retrieve (dùng chung)"):
            for i, c in enumerate(chunks):
                st.markdown(f"**Chunk {i+1}** (score: {c['score']})")
                st.write(c["text"][:300])
                st.divider()