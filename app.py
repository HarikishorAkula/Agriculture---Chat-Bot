import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import hashlib
from pinecone import Pinecone

# ----------------------------

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "agriculture-chatbot"   

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AgriBot 🌾", page_icon="🌱", layout="wide")

# -------------------------------
# CUSTOM CSS — Light theme, fully visible text
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-dark: #1a3a2a;
    --green-mid: #2d6a4f;
    --green-light: #52b788;
    --cream: #f4f7f4;
    --text-main: #1a2e1e;
    --text-soft: #4a6858;
    --border: #c0d8c0;
    --white: #ffffff;
}

/* Force light background everywhere */
html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--cream) !important;
    color: var(--text-main) !important;
}

/* Force all text dark */
p, span, div, label, h1, h2, h3, h4, li {
    color: var(--text-main) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2rem 3rem 2rem !important;
    max-width: 940px;
    margin: auto;
    background: var(--cream) !important;
}

/* HERO */
.hero {
    background: linear-gradient(135deg, #1a3a2a 0%, #2d6a4f 60%, #40916c 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(26,58,42,0.2);
}
.hero::after {
    content: '🌾';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.2;
}
.hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    color: #d8f3dc !important;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 0.72rem;
    margin-bottom: 0.6rem;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    color: #ffffff !important;
    font-size: 2rem;
    margin: 0 0 0.3rem 0;
}
.hero p { color: #b7e4c7 !important; font-size: 0.92rem; margin: 0; }

/* STATS */
.stat-box {
    background: white;
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.stat-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--green-mid) !important;
    line-height: 1;
}
.stat-label {
    font-size: 0.78rem;
    color: var(--text-soft) !important;
    margin-top: 4px;
}

/* QUICK CHIPS */
.stButton > button {
    background: white !important;
    border: 1.5px solid #52b788 !important;
    color: #2d6a4f !important;
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    transition: all 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    background: #2d6a4f !important;
    color: white !important;
    border-color: #2d6a4f !important;
}

/* CHAT BUBBLES */
.farmer-msg {
    background: #ffffff;
    border: 1px solid #d0e8d0;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.4rem;
    margin: 0.8rem 0 0.3rem 0;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.farmer-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #4a6858 !important;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.farmer-text {
    font-size: 0.96rem;
    color: #1a2e1e !important;
    line-height: 1.55;
}

.bot-msg {
    background: linear-gradient(135deg, #edf7f0 0%, #e2f2e8 100%);
    border: 1px solid #a8d8b9;
    border-left: 4px solid #2d6a4f;
    border-radius: 4px 18px 18px 18px;
    padding: 1rem 1.4rem;
    margin: 0.3rem 0 0.8rem 2rem;
    box-shadow: 0 2px 10px rgba(45,106,79,0.08);
}
.bot-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #2d6a4f !important;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.bot-text {
    font-size: 0.96rem;
    color: #1a2e1e !important;
    line-height: 1.65;
}

.cached-tag {
    display: inline-block;
    background: #fff8e1;
    border: 1px solid #f9a825;
    color: #6d4c00 !important;
    font-size: 0.65rem;
    padding: 1px 8px;
    border-radius: 10px;
    margin-left: 8px;
    vertical-align: middle;
}

/* EMPTY STATE */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    background: white;
    border-radius: 18px;
    border: 1.5px dashed var(--border);
    margin-top: 1rem;
}
.empty-state .icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--green-dark) !important;
    margin-bottom: 0.5rem;
    font-size: 1.4rem;
}
.empty-state p { color: var(--text-soft) !important; font-size: 0.9rem; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: #1a3a2a !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #d8f3dc !important;
}
[data-testid="stSidebar"] .stMarkdown strong { color: #ffffff !important; }

/* DIVIDER */
hr { border-color: #d0e8d0 !important; margin: 1rem 0 !important; }

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: white !important;
    color: var(--text-main) !important;
    border: 2px solid var(--green-light) !important;
    border-radius: 14px !important;
}

/* Warning/info box for empty DB */
.db-warning {
    background: #fff8e1;
    border: 1.5px solid #f9a825;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    color: #5d3a00 !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# INIT MODELS — cached once
# -------------------------------
@st.cache_resource
def load_models():
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    genai.configure(api_key=GEMINI_API_KEY)
    gem = genai.GenerativeModel("models/gemini-flash-latest")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(PINECONE_INDEX_NAME)
    return emb, gem, idx

embedding_model, gemini_model, index = load_models()

# -------------------------------
# SESSION STATE
# -------------------------------
for k, v in [("chat_history", []), ("answer_cache", {}), ("api_calls", 0), ("cache_hits", 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------
# FUNCTIONS
# -------------------------------
@st.cache_data(show_spinner=False)
def get_embedding(text):
    return embedding_model.encode(text).tolist()

def get_answer(query):
    key = hashlib.md5(query.strip().lower().encode()).hexdigest()

    # Cache check
    if key in st.session_state.answer_cache:
        return st.session_state.answer_cache[key], True

    # Get embedding
    query_emb = get_embedding(query)

    # Query Pinecone
    results = index.query(vector=query_emb, top_k=3, include_metadata=True)

    context_list = []
    for match in results.get("matches", []):
        if match.get("score", 0) < 0.3:
            continue
        text = match.get("metadata", {}).get("text", "")
        if text:
            context_list.append(text[:400])

    context = "\n".join(context_list)

    # -------------------------------
    # 🔥 STRICT PROMPT (FIXED)
    # -------------------------------
    base_rules = """
STRICT RULES:
- You MUST give exactly 4 bullet points.
- Each bullet must start with "*".
- Each bullet must be on a new line.
- Each bullet must be simple (max 12 words).
- DO NOT give only one point.
- DO NOT write paragraphs.
- End with a new line: Tip: <one simple tip>
"""

    if not context.strip():
        prompt = f"""
You are AgriBot, an expert agriculture assistant for Indian farmers.

{base_rules}

Question:
{query}

Answer:
"""
    else:
        prompt = f"""
You are AgriBot, an expert agriculture assistant.

Use the context below if relevant.

{base_rules}

Context:
{context}

Question:
{query}

Answer:
"""

    # -------------------------------
    # GENERATE RESPONSE
    # -------------------------------
    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=280,
            temperature=0.3
        )
    )

    answer = response.text.strip()

    # -------------------------------
    # 🔥 SAFETY FIX (VERY IMPORTANT)
    # -------------------------------
    bullet_count = answer.count("*")

    if bullet_count < 3:
        retry_prompt = f"""
Give exactly 5 bullet points for the question.

Rules:
- Each bullet starts with "*"
- Short and simple
- Add final line: Tip:

Question: {query}
"""
        retry = gemini_model.generate_content(retry_prompt)
        answer = retry.text.strip()

    # Save in cache
    st.session_state.answer_cache[key] = answer

    return answer, False

# -------------------------------
# CHECK IF INDEX IS EMPTY
# -------------------------------
@st.cache_data(show_spinner=False)
def check_index_stats():
    try:
        stats = index.describe_index_stats()
        return stats.get("total_vector_count", 0)
    except:
        return 0

vector_count = check_index_stats()

# -------------------------------
# HERO
# -------------------------------
st.markdown("""
<div class="hero">
    <div class="badge">⚡ Free-Tier Optimized · AI Powered</div>
    <h1>AgriBot — Farm Assistant</h1>
    <p>Expert advice on crops, diseases, pests, fertilizers &amp; farming methods</p>
</div>
""", unsafe_allow_html=True)

# Show warning if DB is empty
if vector_count == 0:
    st.markdown("""
    <div class="db-warning">
    ⚠️ <strong>Your Pinecone database is empty.</strong> AgriBot will still answer using Gemini's farming knowledge,
    but for best results run <code>upload_data.py</code> to load agriculture data into Pinecone.
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# STATS ROW
# -------------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{len(st.session_state.chat_history)}</div><div class="stat-label">💬 Questions</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.api_calls}</div><div class="stat-label">🔗 API Calls</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.cache_hits}</div><div class="stat-label">⚡ Cache Hits</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-box"><div class="stat-num">{vector_count}</div><div class="stat-label">🗄️ DB Vectors</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# QUICK CHIPS
# -------------------------------
st.markdown("**Quick questions — click to ask instantly:**")
quick_qs = [
    ("🌾", "Rice blast disease treatment"),
    ("🐛", "How to control aphids organically"),
    ("💧", "Best irrigation for dry season"),
    ("🌿", "Natural fertilizer for vegetables"),
    ("🍅", "Tomato leaf curl virus cure"),
]
cols = st.columns(len(quick_qs))
for col, (emoji, q) in zip(cols, quick_qs):
    with col:
        if st.button(f"{emoji} {q}", key=q, use_container_width=True):
            with st.spinner("🌱 Thinking..."):
                ans, cached = get_answer(q)
            if cached:
                st.session_state.cache_hits += 1
            else:
                st.session_state.api_calls += 1
            st.session_state.chat_history.append((q, ans, cached))
            st.rerun()

st.divider()

# -------------------------------
# CHAT INPUT
# -------------------------------
user_input = st.chat_input("Type your farming question here... 🌾")
if user_input:
    with st.spinner("🌱 AgriBot is thinking..."):
        answer, from_cache = get_answer(user_input)
    if from_cache:
        st.session_state.cache_hits += 1
    else:
        st.session_state.api_calls += 1
    st.session_state.chat_history.append((user_input, answer, from_cache))
    st.rerun()

# -------------------------------
# CHAT DISPLAY
# -------------------------------
if not st.session_state.chat_history:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🌱</div>
        <h3>Welcome to AgriBot!</h3>
        <p>Ask me anything about farming — diseases, pests, fertilizers, or crops.<br>
        Use the quick buttons above or type your question below.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for q, a, cached in reversed(st.session_state.chat_history):
        cached_html = '<span class="cached-tag">⚡ cached</span>' if cached else ''
        # Escape for safe HTML display
        q_safe = q.replace("<", "&lt;").replace(">", "&gt;")
        a_safe = a.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        st.markdown(f"""
        <div class="farmer-msg">
            <div class="farmer-label">👨‍🌾 You Asked</div>
            <div class="farmer-text">{q_safe}</div>
        </div>
        <div class="bot-msg">
            <div class="bot-label">🌱 AgriBot {cached_html}</div>
            <div class="bot-text">{a_safe}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# -------------------------------
# SIDEBAR — Helpful farming charts
# -------------------------------
with st.sidebar:
    st.markdown("## 🌾 AgriBot")
    st.markdown("---")

    # Crop season chart
    st.markdown("### 📅 Crop Seasons (India)")
    crop_data = {
        "Kharif (Jun–Oct)": ["Rice", "Cotton", "Maize", "Soybean", "Groundnut"],
        "Rabi (Nov–Apr)": ["Wheat", "Barley", "Mustard", "Chickpea", "Peas"],
        "Zaid (Apr–Jun)": ["Watermelon", "Cucumber", "Bitter gourd", "Pumpkin"],
    }
    for season, crops in crop_data.items():
        with st.expander(season):
            for c in crops:
                st.markdown(f"• {c}")

    st.markdown("---")

    # Fertilizer NPK guide
    st.markdown("### 🧪 NPK Quick Guide")
    npk_data = [
        ("Leafy growth", "High N", "Urea, DAP"),
        ("Root strength", "High P", "SSP, DAP"),
        ("Fruit/flower", "High K", "MOP, SOP"),
        ("All-round", "Balanced", "NPK 19-19-19"),
    ]
    for purpose, ratio, source in npk_data:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1);border-radius:10px;padding:8px 10px;margin-bottom:6px;">
        <strong style="color:#d8f3dc!important">{purpose}</strong><br>
        <span style="color:#b7e4c7!important;font-size:0.82rem">{ratio} · {source}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Common pests quick reference
    st.markdown("### 🐛 Pest Quick Reference")
    pests = [
        ("Aphids", "Neem oil spray"),
        ("Bollworm", "Bt spray / Pheromone trap"),
        ("Whitefly", "Yellow sticky traps"),
        ("Thrips", "Blue sticky traps"),
        ("Mealybug", "Neem + soap solution"),
    ]
    for pest, remedy in pests:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:6px 10px;margin-bottom:5px;font-size:0.82rem;">
        🐛 <strong style="color:#ffffff!important">{pest}</strong><br>
        <span style="color:#b7e4c7!important">→ {remedy}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Soil pH guide
    st.markdown("### 🌍 Soil pH Guide")
    ph_guide = [
        ("4.5–5.5", "Tea, Potato, Blueberry"),
        ("5.5–6.5", "Rice, Maize, Tomato"),
        ("6.5–7.5", "Wheat, Cotton, Onion"),
        ("7.5–8.5", "Barley, Spinach, Beet"),
    ]
    for ph, crops in ph_guide:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.08);border-radius:8px;padding:6px 10px;margin-bottom:5px;font-size:0.82rem;">
        <strong style="color:#d4a843!important">pH {ph}</strong><br>
        <span style="color:#b7e4c7!important">{crops}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Pinecone · Gemini 1.5 Flash · HuggingFace")