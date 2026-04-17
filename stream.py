import streamlit as st
import os

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="🌾 AgriBot", page_icon="🌱", layout="wide")
st.title("🌾 Smart Agriculture Assistant (Pinecone Only)")
st.write("Answers strictly from your uploaded agriculture documents")

# ----------------------------
# MODEL
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# PINECONE
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("agriculture-chatbot")

# ----------------------------
# EMBEDDING
# ----------------------------
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# ----------------------------
# STRICT PINECONE ANSWER
# ----------------------------
def get_answer(query):

    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    matches = results.get("matches", [])

    # filter low-score results
    filtered = [m for m in matches if m.get("score", 0) > 0.6]

    if not filtered:
        return "❌ Not found in agriculture documents."

    context_list = []
    for m in filtered:
        text = m.get("metadata", {}).get("text", "")
        if text:
            context_list.append(text)

    if not context_list:
        return "❌ No valid context found in documents."

    # STRICT OUTPUT (NO REWRITING)
    return "\n\n".join(context_list[:3])

# ----------------------------
# CHAT MEMORY
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ----------------------------
# INPUT
# ----------------------------
query = st.chat_input("Ask your farming question 🌾")

if query:
    response = get_answer(query)
    st.session_state.chat.append((query, response))

# ----------------------------
# DISPLAY CHAT
# ----------------------------
for q, r in reversed(st.session_state.chat):

    st.markdown("### 👨‍🌾 Farmer")
    st.info(q)

    st.markdown("### 🌱 AgriBot (From Documents Only)")
    st.success(r)

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("📌 System Rules")
    st.write("""
    ✔ Only Pinecone data used  
    ✔ No external AI generation  
    ✔ No hallucinated answers  
    ✔ Returns only stored document text  
    """)