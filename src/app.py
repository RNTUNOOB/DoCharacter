import streamlit as st
import json
import os
import shutil
from chat_engine import ChatController
# Import our new pipeline module
from ingest_pipeline import ingest_new_book

# --- CONFIGURATION ---
DATA_DIR = "data"
# ---------------------

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

# Initialize uploader key if not present
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar.expander("📥 Import New Book", expanded=False):
    # Use the dynamic key
    uploaded_file = st.file_uploader(
        "Upload PDF/EPUB", 
        type=["pdf", "epub", "txt"], 
        key=f"uploader_{st.session_state.uploader_key}"
    )
    # Also key the text input so we can clear it
    book_name_input = st.text_input(
        "Book Title (Short)", 
        placeholder="e.g. Harry_Potter",
        key=f"book_name_{st.session_state.uploader_key}"
    )
    
    if st.button("Start Ingestion"):
        if uploaded_file and book_name_input:
            with st.spinner("Processing Book... Watch terminal for details..."):
                try:
                    clean_name = ingest_new_book(uploaded_file, book_name_input)
                    st.success(f"✅ Import Complete! {clean_name} added.")
                    
                    # Increment key to force reset of widgets
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

# --- LOAD RESOURCES ---
controller = load_engine()

# Load Characters
characters = {}
if os.path.exists(char_file):
    try:
        with open(char_file, 'r') as f:
            characters = json.load(f)
    except:
        st.sidebar.error("Character file corrupt.")

# Load Vector DB
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

if os.path.exists(timeline_path):
    with open(timeline_path, 'r') as f: timeline_data = json.load(f)
    
    total_arcs = len(timeline_data)
    if total_arcs > 0:
        selected_arc_idx = st.sidebar.slider(
            "Story Progress", 1, total_arcs, 1, format="Arc %d"
        )
        
        arc_data = timeline_data[selected_arc_idx - 1]
        st.sidebar.caption(f"_{arc_data.get('summary', 'No summary')}_")
        
        # Get characters in this arc
        raw_arc_chars = arc_data.get("characters_present", [])
        # Fuzzy match against our profile DB
        for char_key in characters.keys():
            # Create a set of name parts for the known character (e.g., {"sherlock", "holmes"})
            # Filter out short words to avoid matching "The", "Mr", etc.
            char_parts = set(part.lower() for part in char_key.split() if len(part) > 2)
            
            match_found = False
            for raw_name in raw_arc_chars:
                # Create a set for the raw name found in timeline (e.g., {"mr", "holmes"})
                raw_parts = set(part.lower() for part in raw_name.split() if len(part) > 2)
                
                # 1. Check exact substring (Original logic - kept for safety)
                if char_key.lower() in raw_name.lower() or raw_name.lower() in char_key.lower():
                    match_found = True
                
                # 2. Check Token Intersection (The Fix: matches "Holmes" to "Sherlock Holmes")
                elif char_parts.intersection(raw_parts):
                    match_found = True
                
                if match_found:
                    if char_key not in current_arc_chars:
                        current_arc_chars.append(char_key)
                    break
        
        # Build Context for AI
        char_moods = arc_data.get("moods", {})
        current_mood_context = f"TIMELINE (Arc {selected_arc_idx}): {arc_data.get('summary')}\nMOODS: {char_moods}"

# --- CAST DISPLAY (Dynamic) ---
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

selected_char = st.sidebar.radio(
    "Select Character:", 
    all_chars_sorted, 
    format_func=char_formatter, 
    label_visibility="collapsed"
)

if selected_char:
    char_data = characters[selected_char]
    
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
    
    # Show mood if in current arc
    if selected_char in current_arc_chars:
        # Try to find specific mood
        mood = "Present"
        # Basic lookup in the mood dictionary
        if 'char_moods' in locals():
            for k, v in char_moods.items():
                if selected_char.lower() in k.lower():
                    mood = v
                    break
        st.sidebar.info(f"**Current Status:** {mood}")

# --- MAIN CHAT ---
st.title(f"{selected_book_name.replace('_', ' ')}")

session_key = f"{selected_book_name}_{selected_char}"
if "messages" not in st.session_state: st.session_state.messages = {}
if session_key not in st.session_state.messages: st.session_state.messages[session_key] = []

# History
for message in st.session_state.messages[session_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input(f"Message {selected_char}..."):
    st.session_state.messages[session_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(f"{selected_char} is thinking..."):
            
            # Get current history (excluding the pending prompt)
            current_history = st.session_state.messages[session_key]
            
            response = controller.chat(
                user_query=prompt, 
                char_name=selected_char, 
                char_data=char_data, 
                current_arc_context=current_mood_context, 
                history=current_history,
                all_characters_data=characters,     # <--- Passes all profiles
                full_timeline_data=timeline_data    # <--- Passes full timeline
            )
            st.markdown(response)
    
    st.session_state.messages[session_key].append({"role": "assistant", "content": response})

if st.sidebar.button("Clear History"):
    st.session_state.messages[session_key] = []
    st.rerun()