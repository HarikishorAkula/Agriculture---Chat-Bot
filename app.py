import streamlit as st
from sentence_transformers import SentenceTransformer
import hashlib
from pinecone import Pinecone
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "agriculture-chatbot"

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AgriBot 🌾", page_icon="🌱", layout="wide")

# -------------------------------
# CUSTOM CSS
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

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--cream) !important;
    color: var(--text-main) !important;
}

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
.doc-tag {
    display: inline-block;
    background: #e3f2fd;
    border: 1px solid #1976d2;
    color: #0d47a1 !important;
    font-size: 0.65rem;
    padding: 1px 8px;
    border-radius: 10px;
    margin-left: 8px;
    vertical-align: middle;
}

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

hr { border-color: #d0e8d0 !important; margin: 1rem 0 !important; }

[data-testid="stChatInput"] textarea {
    background: white !important;
    color: var(--text-main) !important;
    border: 2px solid var(--green-light) !important;
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# SESSION STATE
# -------------------------------
for k, v in [
    ("chat_history", []),
    ("answer_cache", {}),
    ("api_calls", 0),
    ("cache_hits", 0),
    ("gemini_failed", False),
    ("force_pinecone_only", False),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------------------
# MODE HELPER
# -------------------------------
def is_pinecone_only():
    return st.session_state.force_pinecone_only or st.session_state.gemini_failed

# -------------------------------
# INIT MODELS
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def load_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gem = genai.GenerativeModel("models/gemini-flash-latest")
        return gem
    except Exception:
        return None

embedding_model = load_embedding_model()
index = load_pinecone_index()
gemini_model = load_gemini() if GEMINI_API_KEY else None

# -------------------------------
# FUNCTIONS
# -------------------------------
@st.cache_data(show_spinner=False)
def get_embedding(text):
    return embedding_model.encode(text).tolist()

def get_answer_pinecone_only(query):
    query_emb = get_embedding(query)
    results = index.query(vector=query_emb, top_k=5, include_metadata=True)
    matches = results.get("matches", [])
    filtered = [m for m in matches if m.get("score", 0) > 0.45]

    if not filtered:
        return "❌ Not found in agriculture documents. Try rephrasing your question."

    context_list = []
    for m in filtered:
        text = m.get("metadata", {}).get("text", "")
        if text:
            context_list.append(text.strip())

    if not context_list:
        return "❌ No valid context found in documents."

    return "\n\n---\n\n".join(context_list[:3])

def get_answer_gemini(query):
    key = hashlib.md5(query.strip().lower().encode()).hexdigest()

    if key in st.session_state.answer_cache:
        return st.session_state.answer_cache[key], True, False

    query_emb = get_embedding(query)
    results = index.query(vector=query_emb, top_k=3, include_metadata=True)

    context_list = []
    for match in results.get("matches", []):
        if match.get("score", 0) < 0.3:
            continue
        text = match.get("metadata", {}).get("text", "")
        if text:
            context_list.append(text[:400])
    context = "\n".join(context_list)

    base_rules = """
STRICT RULES:
- Give exactly 4 to 5 bullet points.
- Each bullet starts with "*".
- Each bullet on a new line.
- Each bullet max 15 words.
- End with: Tip: <one simple tip>
"""
    if not context.strip():
        prompt = f"You are AgriBot, an expert agriculture assistant for Indian farmers.\n{base_rules}\nQuestion:\n{query}\nAnswer:"
    else:
        prompt = f"You are AgriBot, an expert agriculture assistant.\nUse the context below if relevant.\n{base_rules}\nContext:\n{context}\nQuestion:\n{query}\nAnswer:"

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=280, temperature=0.3)
        )
        answer = response.text.strip()

        if answer.count("*") < 3:
            retry = gemini_model.generate_content(
                f"Give exactly 5 bullet points for: {query}\nRules: each starts with *, short, add Tip: at end."
            )
            answer = retry.text.strip()

        st.session_state.answer_cache[key] = answer
        return answer, False, False

    except Exception as e:
        err_str = str(e).lower()
        if any(x in err_str for x in ["quota", "429", "resource_exhausted", "rate limit", "billing"]):
            st.session_state.gemini_failed = True
            fallback = get_answer_pinecone_only(query)
            return fallback, False, True
        else:
            return f"⚠️ Gemini error: {str(e)}", False, False

def get_answer(query):
    key = hashlib.md5(query.strip().lower().encode()).hexdigest()

    if is_pinecone_only():
        if key in st.session_state.answer_cache:
            return st.session_state.answer_cache[key], True, "pinecone"
        answer = get_answer_pinecone_only(query)
        st.session_state.answer_cache[key] = answer
        return answer, False, "pinecone"
    else:
        answer, cached, gemini_failed = get_answer_gemini(query)
        mode = "pinecone" if gemini_failed else "gemini"
        return answer, cached, mode

# -------------------------------
# CHECK INDEX STATS
# -------------------------------
@st.cache_data(show_spinner=False)
def check_index_stats():
    try:
        stats = index.describe_index_stats()
        return stats.get("total_vector_count", 0)
    except:
        return 0

vector_count = check_index_stats()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("## 🌾 AgriBot")
    st.markdown("---")

    st.markdown("### ⚙️ Answer Mode")

    if is_pinecone_only():
        st.markdown("""
        <div style="background:rgba(249,168,37,0.15);border:1px solid #f9a825;border-radius:10px;padding:10px;margin-bottom:10px;">
        <strong style="color:#ffd54f!important">📄 Pinecone-Only Mode</strong><br>
        <span style="color:#ffe082!important;font-size:0.8rem">Answers come directly from your uploaded documents. No AI generation.</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ Switch to Gemini + Pinecone", use_container_width=True):
            st.session_state.force_pinecone_only = False
            st.session_state.gemini_failed = False
            st.rerun()
    else:
        if st.button("📄 Switch to Pinecone-Only (No Gemini)", use_container_width=True):
            st.session_state.force_pinecone_only = True
            st.rerun()

    if st.session_state.gemini_failed:
        st.markdown("""
        <div style="background:rgba(229,57,53,0.15);border:1px solid #ef5350;border-radius:8px;padding:8px;margin-top:6px;">
        <span style="color:#ff8a80!important;font-size:0.8rem">⚠️ Gemini quota exhausted. Auto-switched to Pinecone-only mode.</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.answer_cache = {}
        st.success("Cache cleared!")

    st.markdown("---")

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
    st.caption("Pinecone · Gemini · HuggingFace")

# ===============================
# MAIN CONTENT
# ===============================

mode_badge = "📄 Pinecone-Only Mode" if is_pinecone_only() else "🤖 Gemini + Pinecone · AI Powered"
st.markdown(f"""
<div class="hero">
    <div class="badge">⚡ {mode_badge}</div>
    <h1>Hari's AgriBot — Farm Assistant</h1>
    <p>Expert advice on crops, diseases, pests, fertilizers &amp; farming methods</p>
</div>
""", unsafe_allow_html=True)

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
                ans, cached, mode = get_answer(q)
            if cached:
                st.session_state.cache_hits += 1
            else:
                st.session_state.api_calls += 1
            st.session_state.chat_history.append((q, ans, cached, mode))
            st.rerun()

st.divider()

user_input = st.chat_input("Type your farming question here... 🌾")
if user_input:
    with st.spinner("🌱 AgriBot is thinking..."):
        answer, from_cache, mode = get_answer(user_input)
    if from_cache:
        st.session_state.cache_hits += 1
    else:
        st.session_state.api_calls += 1
    st.session_state.chat_history.append((user_input, answer, from_cache, mode))
    st.rerun()

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
    for item in reversed(st.session_state.chat_history):
        if len(item) == 4:
            q, a, cached, mode = item
        else:
            q, a, cached = item
            mode = "gemini"

        cached_html = '<span class="cached-tag">⚡ cached</span>' if cached else ''
        mode_html = '<span class="doc-tag">📄 from docs</span>' if mode == "pinecone" else ''

        q_safe = q.replace("<", "&lt;").replace(">", "&gt;")
        a_safe = a.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

        st.markdown(f"""
        <div class="farmer-msg">
            <div class="farmer-label">👨‍🌾 You Asked</div>
            <div class="farmer-text">{q_safe}</div>
        </div>
        <div class="bot-msg">
            <div class="bot-label">🌱 AgriBot {cached_html}{mode_html}</div>
            <div class="bot-text">{a_safe}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()
