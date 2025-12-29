import streamlit as st
import json
import os
import graphviz
import time

# =========================================================
# STATIC / NULL CONTROLLER (NO LLMs)
# =========================================================


class StaticChatController:
    def load_book_resources(self, *args, **kwargs):
        return None

    def unload_resources(self):
        return None

    def chat(
        self,
        user_query,
        char_name,
        char_data,
        current_arc_context,
        history,
        all_characters_data,
        full_timeline_data,
        selected_arc_id,
    ):
        return {
            "response": (
                "⚠️ **Static Preview Mode**\n\n"
                "This deployment is running without LLMs.\n\n"
                "The UI is fully interactive, but responses are disabled."
            ),
            "confidence": 0.0,
            "sources": [],
        }


@st.cache_resource
def load_engine():
    return StaticChatController()


# =========================================================
# CONFIG
# =========================================================

DATA_DIR = "data"

st.set_page_config(
    page_title="DoCharacter",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded",
)

st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
st.info("🚧 Running in Static Preview Mode (LLMs disabled)")

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stExpander"] details summary > div { padding-left: 0.5rem; }
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 20px;
        border: 1px solid rgba(49, 51, 63, 0.2);
        padding: 0.25rem 1rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# UTILS
# =========================================================


def get_initials(name):
    return name[:2].upper() if len(name) > 1 else name[:1].upper()


def get_avatar_color(name):
    colors = ["#FF5733", "#33FF57", "#3357FF", "#F033FF", "#FF33A8", "#33FFF5"]
    return colors[sum(ord(c) for c in name) % len(colors)]


def visualize_knowledge_graph(kb_data):
    if not kb_data:
        return None

    graph = graphviz.Digraph()
    graph.attr(rankdir="LR", size="10,6", bgcolor="transparent")
    graph.attr("node", shape="box", style="rounded,filled", fillcolor="lightblue")

    for fact in kb_data[:60]:
        s = str(fact.get("subject", "Unknown"))[:20]
        o = str(fact.get("object", "Unknown"))[:20]
        p = str(fact.get("predicate", "related to"))
        graph.edge(s, o, label=p)

    return graph


# =========================================================
# SIDEBAR – LIBRARY
# =========================================================

with st.sidebar:
    st.header("📚 Library")

    os.makedirs(DATA_DIR, exist_ok=True)

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

    with st.expander("📥 Add New Book"):
        st.file_uploader("Upload PDF/EPUB", type=["pdf", "epub", "txt"])
        st.text_input("Short Title")
        if st.button("Start Ingestion", type="primary", use_container_width=True):
            st.info("📦 Static mode: ingestion disabled.")
            st.toast("Upload acknowledged (no processing).")

    st.divider()
    st.subheader("Actions")

    if st.button("⛏️ Mine Knowledge", use_container_width=True):
        st.info("⛏️ Static mode: mining disabled.")
        st.toast("Using cached knowledge.json (if available).")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.clear()
        st.toast("Chat history cleared.")
        st.rerun()

# =========================================================
# LOAD BOOK DATA
# =========================================================

if not selected_book_name:
    st.info("👈 Upload a book to get started.")
    st.stop()

book_path = os.path.join(DATA_DIR, selected_book_name)
char_file = os.path.join(book_path, "characters.json")
timeline_file = os.path.join(book_path, "timeline.json")
knowledge_file = os.path.join(book_path, "knowledge.json")

characters = {}
timeline_data = []

if os.path.exists(char_file):
    with open(char_file) as f:
        characters = json.load(f)

if os.path.exists(timeline_file):
    with open(timeline_file) as f:
        timeline_data = json.load(f)

controller = load_engine()
controller.load_book_resources(book_path)

# =========================================================
# STORY NAVIGATION
# =========================================================

selected_arc_idx = None
current_arc_chars = []
current_mood_context = ""

with st.sidebar:
    st.divider()
    st.header("🧭 Story Navigation")

    if timeline_data:
        sections = {}
        for e in timeline_data:
            sections.setdefault(e.get("section", "General"), []).append(e)

        selected_section = st.selectbox("Jump to Section:", sections.keys())
        arcs = sections[selected_section]

        arc_map = {
            f"Scene {a['arc_id']}: {a['summary'][:40]}": a["arc_id"] for a in arcs
        }
        selected_scene = st.selectbox("Select Scene:", arc_map.keys())
        selected_arc_idx = arc_map[selected_scene]

        arc_data = next(a for a in timeline_data if a["arc_id"] == selected_arc_idx)
        st.info(arc_data.get("summary", ""))

        current_arc_chars = arc_data.get("characters_present", [])
        current_mood_context = arc_data.get("summary", "")
    else:
        st.caption("No timeline data available.")

# =========================================================
# MAIN CONTENT
# =========================================================

st.title(selected_book_name.replace("_", " "))

tab_chat, tab_chars, tab_graph = st.tabs(
    ["💬 Chat", "👥 Characters", "🕸️ Knowledge Graph"]
)

# =========================================================
# CHAT TAB
# =========================================================

with tab_chat:
    AUDITOR_KEY = "The_Auditor"

    if "active_char_id" not in st.session_state:
        st.session_state.active_char_id = AUDITOR_KEY

    char_options = [AUDITOR_KEY] + [c for c in characters if c != AUDITOR_KEY]

    st.selectbox(
        "Select Character:",
        char_options,
        key="active_char_id",
    )

    selected_char = st.session_state.active_char_id
    char_data = characters.get(selected_char, {})

    chat_container = st.container(height=600)
    session_key = f"{selected_book_name}_{selected_char}"

    st.session_state.setdefault("messages", {})
    st.session_state.messages.setdefault(session_key, [])

    with chat_container:
        for msg in st.session_state.messages[session_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input(f"Message {selected_char}..."):
        st.session_state.messages[session_key].append(
            {"role": "user", "content": prompt}
        )

        response = controller.chat(
            prompt,
            selected_char,
            char_data,
            current_mood_context,
            [],
            characters,
            timeline_data,
            selected_arc_idx,
        )

        st.session_state.messages[session_key].append(
            {"role": "assistant", "content": response["response"]}
        )
        st.rerun()

# =========================================================
# CHARACTERS TAB
# =========================================================

with tab_chars:
    cols = st.columns(3)
    for i, (name, data) in enumerate(characters.items()):
        if name == AUDITOR_KEY:
            continue
        with cols[i % 3]:
            st.markdown(f"### {name}")
            st.caption(f"{data.get('role')} • {data.get('personality')}")
            st.markdown(data.get("bio", "No bio available."))

# =========================================================
# KNOWLEDGE GRAPH TAB
# =========================================================

with tab_graph:
    if os.path.exists(knowledge_file):
        with open(knowledge_file) as f:
            kb = json.load(f)
        graph = visualize_knowledge_graph(kb)
        if graph:
            st.graphviz_chart(graph)
        else:
            st.info("Knowledge base empty.")
    else:
        st.info("No Knowledge Graph found.")
