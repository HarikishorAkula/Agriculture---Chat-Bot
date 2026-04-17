import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="🌾 AgriBot", page_icon="🌱", layout="wide")

st.title("🌾 Smart Agriculture Assistant (FREE AI)")
st.write("Ask anything about crops, diseases, pests, fertilizers")

# ----------------------------
# EMBEDDING MODEL (FREE)
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# PINECONE INIT
# ----------------------------
pc = Pinecone(api_key="pcsk_2u5baP_KxS8L6nqv1W44Z26pSRsWaoZE3sYcqMq54BDKQ1M6qnqRHbYHaih1YFr4zwb8Y3")
index = pc.Index("agriculture-chatbot")

# ----------------------------
# EMBEDDING FUNCTION
# ----------------------------
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# ----------------------------
# PURE CONTEXT ANSWER (NO AI)
# ----------------------------
def get_answer(query):

    query_embedding = get_embedding(query)

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    matches = results.get("matches", [])

    if not matches:
        return "❌ No information found in agriculture database."

    context_list = []
    for m in matches:
        text = m.get("metadata", {}).get("text", "")
        if text:
            context_list.append(text)

    # ----------------------------
    # SMART ANSWER (RULE-BASED)
    # ----------------------------

    answer = "🌾 AGRICULTURE ADVICE:\n\n"

    for i, text in enumerate(context_list[:3]):
        answer += f"{i+1}. {text}\n\n"

    answer += "✔ Based on your agriculture documents."

    return answer

# ----------------------------
# CHAT HISTORY
# ----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# ----------------------------
# INPUT BOX
# ----------------------------
query = st.chat_input("Ask your farming question... 🌾")

if query:
    response = get_answer(query)
    st.session_state.chat.append((query, response))

# ----------------------------
# DISPLAY CHAT
# ----------------------------
for q, r in reversed(st.session_state.chat):

    st.markdown("### 👨‍🌾 Farmer")
    st.info(q)

    st.markdown("### 🌱 AgriBot")
    st.success(r)

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.header("🌾 Features")
    st.write("""
    ✔ Crop disease info  
    ✔ Pest control  
    ✔ Fertilizer guidance  
    ✔ Organic farming tips  
    ✔ 100% FREE (NO API)
    """)