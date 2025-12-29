import streamlit as st
import json
import os
import graphviz
import time
from chat_engine import ChatController
from ingest_pipeline import ingest_new_book
from knowledge_builder import mine_knowledge_graph

# --- CONFIGURATION ---
DATA_DIR = "data"

st.set_page_config(
    page_title="DoCharacter",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stExpander"] details summary > div { padding-left: 0.5rem; }
    
    /* Tab styling */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px;
        border: 1px solid rgba(49, 51, 63, 0.2);
        padding: 0.25rem 1rem;
    }
    
    /* Card styling for Characters tab */
    .char-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444d;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# --- UTILS ---
def get_initials(name):
    return name[:2].upper() if len(name) > 1 else name[:1].upper()


def get_avatar_color(name):
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F033FF", "#FF33A8", "#33FFF5"]
    hash_val = sum(ord(c) for c in name)
    return colors[hash_val % len(colors)]


def visualize_knowledge_graph(kb_data):
    """Generates a Graphviz Digraph from knowledge triples."""
    if not kb_data:
        return None

    graph = graphviz.Digraph()
    graph.attr(rankdir="LR", size="10,6", bgcolor="transparent")
    graph.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fillcolor="lightblue",
        fontname="Helvetica",
    )
    graph.attr("edge", fontname="Helvetica", fontsize="10")

    # Limit to top 60 connections to prevent UI crashing
    for fact in kb_data[:60]:
        subj = str(fact.get("subject", "Unknown"))
        obj = str(fact.get("object", "Unknown"))
        pred = str(fact.get("predicate", "related to"))

        # Shorten labels
        if len(subj) > 20:
            subj = subj[:20] + "..."
        if len(obj) > 20:
            obj = obj[:20] + "..."

        graph.edge(subj, obj, label=pred)

    return graph


@st.cache_resource
def load_engine():
    return ChatController()


# --- SIDEBAR: LIBRARY & ACTIONS ---
with st.sidebar:
    st.header("📚 Library")

    # 1. Book Selector
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    available_books = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    if not available_books:
        st.warning("No books found.")
        selected_book_name = None
    else:
        selected_book_name = st.selectbox(
            "Current Book", available_books, label_visibility="collapsed"
        )

    # 2. Ingestion Expander
    with st.expander("📥 Add New Book"):
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_file = st.file_uploader(
            "Upload PDF/EPUB",
            type=["pdf", "epub", "txt"],
            key=f"uploader_{st.session_state.uploader_key}",
        )
        book_name_input = st.text_input(
            "Short Title",
            placeholder="e.g. Mahabharata",
            key=f"book_name_{st.session_state.uploader_key}",
        )

        if st.button("Start Ingestion", type="primary", use_container_width=True):
            if uploaded_file and book_name_input:
                temp_controller = load_engine()
                temp_controller.unload_resources()

                with st.spinner("Reading & Analyzing..."):
                    try:
                        clean_name = ingest_new_book(uploaded_file, book_name_input)
                        st.success(f"Ready: {clean_name}")
                        st.session_state.uploader_key += 1
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("File and Title required.")

    # 3. Actions (Mining)
    st.divider()
    st.subheader("Actions")
    book_path = os.path.join(DATA_DIR, selected_book_name) if selected_book_name else ""
    knowledge_file = os.path.join(book_path, "knowledge.json")
    kb_exists = os.path.exists(knowledge_file) and os.path.getsize(knowledge_file) > 5

    if st.button(
        "⛏️ Mine Knowledge",
        help="Extract facts for the Auditor",
        use_container_width=True,
    ):
        if selected_book_name:
            with st.spinner("Mining facts..."):
                try:
                    temp_controller = load_engine()
                    temp_controller.unload_resources()
                    success = mine_knowledge_graph(book_path)
                    if success:
                        st.toast("✅ Knowledge Graph Updated!")
                        temp_controller.load_book_resources(book_path)
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        else:
            st.error("Select a book first.")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        # Clear chat for active book
        keys_to_del = [
            k for k in st.session_state.keys() if k.startswith(f"{selected_book_name}_")
        ]
        for k in keys_to_del:
            del st.session_state[k]
        st.toast("Chat history cleared.")
        st.rerun()


# --- LOAD RESOURCES ---
if not selected_book_name:
    st.info("👈 Upload a book to get started.")
    st.stop()

char_file = os.path.join(book_path, "characters.json")
timeline_path = os.path.join(book_path, "timeline.json")

controller = load_engine()

# Load Data Files
characters = {}
if os.path.exists(char_file):
    try:
        with open(char_file, "r") as f:
            characters = json.load(f)
    except Exception:
        pass

timeline_data = []
if os.path.exists(timeline_path):
    try:
        with open(timeline_path, "r") as f:
            timeline_data = json.load(f)
    except Exception:
        pass

try:
    controller.load_book_resources(book_path)
except Exception as e:
    st.sidebar.error(f"DB Error: {e}")

# --- SIDEBAR: STORY NAVIGATION ---
selected_arc_idx = 1
current_mood_context = ""
current_arc_chars = []

with st.sidebar:
    st.divider()  # Visual separator
    st.header("🧭 Story Navigation")

    if timeline_data:
        # Group sections
        sections = {}
        for entry in timeline_data:
            sec = entry.get("section", "General")
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(entry)

        # Section Selector
        selected_section = st.selectbox("Jump to Section:", list(sections.keys()))

        # Scene Selector
        section_arcs = sections[selected_section]
        arc_options = {
            f"Scene {e['arc_id']}: {e['summary'][:50]}...": e["arc_id"]
            for e in section_arcs
        }

        selected_scene_label = st.selectbox("Select Scene:", list(arc_options.keys()))
        selected_arc_idx = arc_options[selected_scene_label]

        # Update Context vars
        arc_data = next(
            (item for item in timeline_data if item["arc_id"] == selected_arc_idx), None
        )
        if arc_data:
            with st.expander("Scene Context", expanded=True):
                st.info(f"_{arc_data.get('summary')}_")

            current_mood_context = f"SECTION: {selected_section}\nSCENE: {arc_data.get('summary')}\nMOODS: {arc_data.get('moods')}"

            # Highlight active characters
            raw_arc_chars = arc_data.get("characters_present", [])
            for item in raw_arc_chars:
                name_str = item if isinstance(item, str) else item.get("name", "")
                for char_key in characters.keys():
                    if (
                        char_key.lower() in name_str.lower()
                        or name_str.lower() in char_key.lower()
                    ):
                        if char_key not in current_arc_chars:
                            current_arc_chars.append(char_key)
    else:
        st.caption("No timeline data available.")

# --- MAIN CONTENT ---
st.title(f"{selected_book_name.replace('_', ' ')}")

# Navigation Tabs
tab_chat, tab_chars, tab_graph = st.tabs(
    ["💬 Chat", "👥 Characters", "🕸️ Knowledge Graph"]
)

# ==========================================
# TAB 1: CHAT INTERFACE
# ==========================================
with tab_chat:
    # --- CHARACTER SELECTION (Fixed Logic) ---
    AUDITOR_KEY = "The_Auditor"

    # 1. Create lists
    scene_chars_sorted = [
        c for c in current_arc_chars if c in characters and c != AUDITOR_KEY
    ]
    other_chars_sorted = [
        c for c in characters.keys() if c not in scene_chars_sorted and c != AUDITOR_KEY
    ]

    # 2. Ensure State Variable exists (but don't bind it to the widget yet)
    if "active_char_id" not in st.session_state:
        st.session_state.active_char_id = AUDITOR_KEY

    # 3. Construct Options (MUST INCLUDE CURRENT SELECTION)
    # Start with Auditor
    char_options = [AUDITOR_KEY]

    # Add Scene Characters
    char_options.extend(scene_chars_sorted)

    # CRITICAL FIX: Check if the user's currently selected character is in the list.
    # If not, force add them. This prevents the widget from crashing or resetting index.
    current_selection = st.session_state.active_char_id
    if current_selection not in char_options and current_selection in characters:
        char_options.append(current_selection)

    # Add remaining characters
    remaining = [c for c in other_chars_sorted if c not in char_options]
    char_options.extend(remaining)

    def format_char_option(name):
        if name == AUDITOR_KEY:
            return "🤖 The Auditor"
        if name == current_selection:
            return f"🔹 {name}"
        if name in scene_chars_sorted:
            return f"📍 {name}"
        return name

    # 4. Render Widget
    # We use `index` to set the initial value based on `active_char_id`
    # We use `on_change` to update `active_char_id` when the user clicks

    def on_char_change():
        # Update the state variable from the widget's value
        st.session_state.active_char_id = st.session_state.char_selector_widget

    try:
        current_index = char_options.index(current_selection)
    except ValueError:
        current_index = 0
        st.session_state.active_char_id = AUDITOR_KEY

    if hasattr(st, "pills"):
        st.pills(
            "Select Character:",
            options=char_options,
            format_func=format_char_option,
            selection_mode="single",
            default=current_selection,  # Use default instead of index for pills
            key="char_selector_widget",
            on_change=on_char_change,
        )
    else:
        st.selectbox(
            "Select Character:",
            options=char_options,
            format_func=format_char_option,
            index=current_index,
            key="char_selector_widget",
            on_change=on_char_change,
        )

    # 5. Use the state variable as the source of truth for the rest of the app
    selected_char = st.session_state.active_char_id

    st.divider()

    # --- CHAT DISPLAY ---
    if selected_char:
        char_data = characters.get(selected_char, {})
        initials = get_initials(selected_char)
        bg_color = get_avatar_color(selected_char)

        # Header
        col1, col2 = st.columns([1, 10])
        with col1:
            st.markdown(
                f"""
            <div style="
                width: 45px; height: 45px;
                background-color: {bg_color};
                border-radius: 50%;
                display: flex; justify-content: center; align-items: center;
                color: white; font-size: 18px; font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                {initials}
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            role = char_data.get("role", "Unknown")
            personality = char_data.get("personality", "Unknown")
            desc = (
                "Objective AI Observer"
                if selected_char == AUDITOR_KEY
                else f"{role} • {personality}"
            )
            st.markdown(
                f"**{selected_char}** \n<span style='color:grey; font-size:0.9em'>{desc}</span>",
                unsafe_allow_html=True,
            )

        # Chat Container (Fixed Height)
        chat_container = st.container(height=700)

        session_key = f"{selected_book_name}_{selected_char}"
        if "messages" not in st.session_state:
            st.session_state.messages = {}
        if session_key not in st.session_state.messages:
            st.session_state.messages[session_key] = []

        with chat_container:
            for message in st.session_state.messages[session_key]:
                with st.chat_message(message["role"]):
                    content = message["content"]
                    if isinstance(content, dict):
                        if "response" in content:
                            st.markdown(content["response"])
                        elif "answer" in content:
                            st.markdown(content["answer"])

                        with st.expander("🧠 Logic & Sources"):
                            if "internal_thought" in content:
                                st.json(content["internal_thought"])
                            if "confidence" in content:
                                st.metric("Confidence", f"{content['confidence']:.2f}")
                            if "sources" in content and content["sources"]:
                                st.markdown("**Sources:**")
                                for src in content["sources"]:
                                    st.caption(f"• {src}")
                    else:
                        st.markdown(content)

        # Input
        if prompt := st.chat_input(f"Message {selected_char}..."):
            st.session_state.messages[session_key].append(
                {"role": "user", "content": prompt}
            )

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        current_history = st.session_state.messages[session_key]
                        response_data = controller.chat(
                            user_query=prompt,
                            char_name=selected_char,
                            char_data=char_data,
                            current_arc_context=current_mood_context,
                            history=current_history,
                            all_characters_data=characters,
                            full_timeline_data=timeline_data,
                            selected_arc_id=selected_arc_idx,
                        )
                        print(response_data)

                        if "response" in response_data:
                            st.markdown(response_data["response"])
                        elif "answer" in response_data:
                            st.markdown(response_data["answer"])

                        with st.expander("🧠 Logic & Sources"):
                            if "internal_thought" in response_data:
                                st.json(response_data["internal_thought"])
                            if "confidence" in response_data:
                                st.metric(
                                    "Confidence", f"{response_data['confidence']:.2f}"
                                )
                            if "sources" in response_data and response_data["sources"]:
                                st.markdown("**Sources:**")
                                for src in response_data["sources"]:
                                    st.caption(f"• {src}")

            st.session_state.messages[session_key].append(
                {"role": "assistant", "content": response_data}
            )
            st.rerun()

# ==========================================
# TAB 2: CHARACTERS (New Feature)
# ==========================================
with tab_chars:
    st.subheader("👥 Dramatis Personae")

    cols = st.columns(3)
    for i, (name, data) in enumerate(characters.items()):
        if name == AUDITOR_KEY:
            continue

        with cols[i % 3]:
            with st.container(border=True):
                # Initials avatar
                initials = get_initials(name)
                bg = get_avatar_color(name)
                st.markdown(
                    f"""
                <div style="display:flex; align-items:center; margin-bottom:10px;">
                    <div style="width:40px; height:40px; background-color:{bg}; border-radius:50%; color:white; display:flex; justify-content:center; align-items:center; font-weight:bold; margin-right:10px;">
                        {initials}
                    </div>
                    <div style="font-weight:bold; font-size:1.1em;">{name}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.caption(
                    f"**Role:** {data.get('role')} | **Type:** {data.get('personality')}"
                )
                st.markdown(f"_{data.get('bio', 'No bio available.')}_")

                if data.get("sample_quotes"):
                    with st.expander("Quotes"):
                        for q in data["sample_quotes"]:
                            if isinstance(q, dict):
                                st.caption(f'"{q.get("text")}"')
                            else:
                                st.caption(f'"{q}"')

# ==========================================
# TAB 3: KNOWLEDGE GRAPH
# ==========================================
with tab_graph:
    if kb_exists:
        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                kb_data = json.load(f)

            if kb_data:
                st.caption(
                    f"Visualizing {min(len(kb_data), 60)} entities from {len(kb_data)} total facts."
                )

                graph = visualize_knowledge_graph(kb_data)
                st.graphviz_chart(graph, use_container_width=True)

                with st.expander("📂 View Raw Fact Data"):
                    st.dataframe(kb_data)
            else:
                st.info("Knowledge base is empty. Use the tool above to build it.")
        except Exception as e:
            st.error(f"Failed to load graph: {e}")
    else:
        st.info("No Knowledge Graph found. Mine it from the sidebar actions.")
