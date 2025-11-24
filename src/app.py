import streamlit as st
import json
import os
from chat_engine import ChatController
from ingest_pipeline import ingest_new_book
from knowledge_builder import mine_knowledge_graph

# --- CONFIGURATION ---
DATA_DIR = "data"

st.set_page_config(page_title="DOCharacter", layout="wide", page_icon="📚")

# --- UTILS ---
def get_initials(name):
    return name[:2].upper() if len(name) > 1 else name[:1].upper()

def get_avatar_color(name):
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F033FF", "#FF33A8", "#33FFF5"]
    hash_val = sum(ord(c) for c in name)
    return colors[hash_val % len(colors)]

@st.cache_resource
def load_engine():
    return ChatController()

# --- SIDEBAR: IMPORT ---
st.sidebar.title("📚 Library")

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar.expander("📥 Import New Book", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload PDF/EPUB",
        type=["pdf", "epub", "txt"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    book_name_input = st.text_input(
        "Book Title (Short)",
        placeholder="e.g. Harry_Potter",
        key=f"book_name_{st.session_state.uploader_key}"
    )
    if st.button("Start Ingestion"):
        if uploaded_file and book_name_input:
            # --- FIX: Release locks before touching the files ---
            # We must load the engine instance to call unload, even if not fully init yet
            temp_controller = load_engine()
            temp_controller.unload_resources()
            
            with st.spinner("Processing Book... Watch terminal for details..."):
                try:
                    clean_name = ingest_new_book(uploaded_file, book_name_input)
                    st.success(f"✅ Import Complete! {clean_name} added.")
                    st.session_state.uploader_key += 1
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {str(e)}")
        else:
            st.warning("Please upload a file and give it a name.")

# --- SIDEBAR: SELECTOR ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

available_books = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
if not available_books:
    st.info("👈 No books found. Please import one in the sidebar!")
    st.stop()

selected_book_name = st.sidebar.selectbox("Select Book", available_books)
book_path = os.path.join(DATA_DIR, selected_book_name)
char_file = os.path.join(book_path, "characters.json")
knowledge_file = os.path.join(book_path, "knowledge.json")

# --- KNOWLEDGE MINER BUTTON ---
kb_exists = os.path.exists(knowledge_file) and os.path.getsize(knowledge_file) > 5
if st.sidebar.button(f"⛏️ {'Update' if kb_exists else 'Build'} Knowledge Graph"):
    with st.spinner("Mining facts... This may take a few minutes..."):
        try:
            # Unload DB to be safe
            temp_controller = load_engine()
            temp_controller.unload_resources()
            
            success = mine_knowledge_graph(book_path)
            if success:
                st.sidebar.success("✅ Knowledge Graph Updated!")
                # Force reload of resources
                temp_controller.load_book_resources(book_path)
        except Exception as e:
            st.sidebar.error(f"Mining failed: {e}")

# Check status for UI
if kb_exists:
    try:
        with open(knowledge_file, 'r') as f:
            fact_count = len(json.load(f))
        st.sidebar.caption(f"🧠 Knowledge Base: {fact_count} facts")
    except:
        st.sidebar.caption("🧠 Knowledge Base: Corrupt")
else:
    st.sidebar.caption("🧠 Knowledge Base: Empty (Auditor limited)")

# Load the controller for Chat
controller = load_engine()

characters = {}
if os.path.exists(char_file):
    try:
        with open(char_file, 'r') as f:
            characters = json.load(f)
    except Exception as e:
        st.sidebar.error("Character file corrupt.")
        characters = {}

try:
    controller.load_book_resources(book_path)
except Exception as e:
    st.sidebar.warning(f"Memory not ready: {e}")

# --- CHARACTER & TIMELINE ---
st.sidebar.divider()
st.sidebar.subheader("⏳ Narrative Arc")

timeline_path = os.path.join(book_path, "timeline.json")
current_arc_chars = []
current_mood_context = ""

timeline_data = []
if os.path.exists(timeline_path):
    try:
        with open(timeline_path, 'r') as f:
            timeline_data = json.load(f)
    except Exception as e:
        st.sidebar.warning("Timeline file corrupt.")
        timeline_data = []

# Initialize selected_arc_idx with a default (Arc 1)
selected_arc_idx = 1

if timeline_data:
    total_arcs = len(timeline_data)
    if total_arcs > 0:
        # --- FIX: Handle single-arc stories to prevent slider crash ---
        if total_arcs > 1:
            selected_arc_idx = st.sidebar.slider("Story Progress", 1, total_arcs, 1, format="Arc %d")
        else:
            # If only 1 arc, force index 1 and show a static label
            selected_arc_idx = 1
            st.sidebar.markdown("**Story Progress:** Arc 1 (Complete)")
        
        arc_data = timeline_data[selected_arc_idx - 1]
        st.sidebar.caption(f"_{arc_data.get('summary', 'No summary')}_")

        raw_arc_chars = arc_data.get("characters_present", [])
        for item in raw_arc_chars:
            if isinstance(item, dict) and 'name' in item:
                raw_name = item['name']
            elif isinstance(item, str):
                raw_name = item
            else:
                continue
            raw_parts = set(part.lower() for part in raw_name.split() if len(part) > 2)
            for char_key in characters.keys():
                char_parts = set(part.lower() for part in char_key.split() if len(part) > 2)
                match_found = False
                if char_key.lower() in raw_name.lower() or raw_name.lower() in char_key.lower():
                    match_found = True
                elif char_parts.intersection(raw_parts):
                    match_found = True
                if match_found:
                    if char_key not in current_arc_chars:
                        current_arc_chars.append(char_key)
                    break

        char_moods_raw = arc_data.get("moods", {})
        char_moods = {}
        if isinstance(char_moods_raw, list):
            for entry in char_moods_raw:
                if isinstance(entry, dict) and 'char' in entry and 'emotion' in entry:
                    char_moods[entry['char']] = entry['emotion']
                # ignore string entries
        elif isinstance(char_moods_raw, dict):
            char_moods = char_moods_raw

        current_mood_context = f"TIMELINE (Arc {selected_arc_idx}): {arc_data.get('summary')}\nMOODS: {char_moods}"

# --- CAST DISPLAY ---
st.sidebar.subheader("🎭 Cast")

AUDITOR_KEY = "The_Auditor"
scene_chars = [c for c in current_arc_chars if c in characters and c != AUDITOR_KEY]
other_chars = [c for c in characters.keys() if c not in scene_chars and c != AUDITOR_KEY]
all_chars_sorted = [AUDITOR_KEY] + scene_chars + other_chars

def char_formatter(name):
    if name == AUDITOR_KEY:
        return "🤖 The Auditor"
    if name in scene_chars:
        return f"📍 {name}"
    return f"👥 {name}"

selected_char = st.sidebar.radio("Select Character:", all_chars_sorted, format_func=char_formatter, label_visibility="collapsed")

if selected_char:
    char_data = characters.get(selected_char, {})
    initials = get_initials(selected_char)
    bg_color = get_avatar_color(selected_char)

    avatar_html = f"""
    <div style="
        width: 80px; height: 80px;
        background-color: {bg_color};
        border-radius: 50%;
        display: flex; justify-content: center; align-items: center;
        color: white; font-size: 30px; font-weight: bold;
        margin: 10px auto 5px auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        {initials}
    </div>
    <div style="text-align: center; font-weight: bold; font-size: 18px; margin-bottom: 10px;">{selected_char}</div>
    """
    st.sidebar.markdown(avatar_html, unsafe_allow_html=True)

    if selected_char in current_arc_chars and 'char_moods' in locals():
        mood = "Present"
        for k, v in char_moods.items():
            if selected_char.lower() in k.lower():
                if isinstance(v, list):
                    mood = ", ".join(v)
                else:
                    mood = v
                break
        st.sidebar.info(f"**Current Status:** {mood}")

# --- MAIN CHAT ---
st.title(f"{selected_book_name.replace('_', ' ')}")

session_key = f"{selected_book_name}_{selected_char}"
if "messages" not in st.session_state:
    st.session_state.messages = {}
if session_key not in st.session_state.messages:
    st.session_state.messages[session_key] = []

for message in st.session_state.messages[session_key]:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, dict):
            if "response" in content:
                st.markdown(content["response"])
            elif "answer" in content:
                st.markdown(content["answer"])
            with st.expander("🧠 Thought Process & Sources"):
                if "internal_thought" in content:
                    st.json(content["internal_thought"])
                if "confidence" in content:
                    st.metric("Confidence", content["confidence"])
                if "sources" in content and content["sources"]:
                    st.write("**Sources:**")
                    for src in content["sources"]:
                        st.caption(f"- {src}")
        else:
            st.markdown(content)

if prompt := st.chat_input(f"Message {selected_char}..."):
    st.session_state.messages[session_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"{selected_char} is thinking..."):
            current_history = st.session_state.messages[session_key]
            response_data = controller.chat(
                user_query=prompt,
                char_name=selected_char,
                char_data=char_data,
                current_arc_context=current_mood_context,
                history=current_history,
                all_characters_data=characters,
                full_timeline_data=timeline_data,
                selected_arc_id=selected_arc_idx  # <-- PASSING THE ID HERE
            )
            if "response" in response_data:
                st.markdown(response_data["response"])
            elif "answer" in response_data:
                st.markdown(response_data["answer"])
            with st.expander("🧠 Thought Process & Sources"):
                if "internal_thought" in response_data:
                    st.json(response_data["internal_thought"])
                if "confidence" in response_data:
                    st.metric("Confidence", response_data["confidence"])
                if "sources" in response_data and response_data["sources"]:
                    st.write("**Sources:**")
                    for src in response_data["sources"]:
                        st.caption(f"- {src}")

    st.session_state.messages[session_key].append({"role": "assistant", "content": response_data})

if st.sidebar.button("Clear History"):
    st.session_state.messages[session_key] = []
    st.rerun()