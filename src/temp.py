# streamlit_app_improved.py
# Improved Streamlit layout for the user's local LLM character-simulation project.
# This file is designed to integrate with the existing project files that were
# uploaded at these local paths:
# - /mnt/data/app.py
# - /mnt/data/chat_engine.py
# - /mnt/data/ingest_pipeline.py
# - /mnt/data/knowledge_builder.py
#
# The code below builds a cleaner UI layout, safer imports of the existing
# modules (via importlib and path loading), and a non-blocking UX by running
# long-running ingestion/knowledge jobs in background threads while streaming
# progress to the UI.

import streamlit as st
import sys
import os
import importlib.util
import threading
import time
import traceback
from typing import Optional

# Paths to user-uploaded modules (these come from your uploaded files)
USER_APP_PATH = "src/app.py"
CHAT_ENGINE_PATH = "../chat_engine.py"
INGEST_PIPELINE_PATH = "../ingest_pipeline.py"
KNOWLEDGE_BUILDER_PATH = "/knowledge_builder.py"

# Utility to import a module from a file path safely
def import_module_from_path(name: str, path: str):
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
        return module
    except Exception:
        st.session_state.setdefault("import_errors", {})[name] = traceback.format_exc()
        return None

# Import modules (if present)
chat_engine_mod = import_module_from_path("chat_engine_uploaded", CHAT_ENGINE_PATH)
ingest_mod = import_module_from_path("ingest_pipeline_uploaded", INGEST_PIPELINE_PATH)
kb_mod = import_module_from_path("knowledge_builder_uploaded", KNOWLEDGE_BUILDER_PATH)
# we don't directly import /mnt/data/app.py to avoid circular replacement; this file replaces it

# Session-state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_book" not in st.session_state:
    st.session_state.selected_book = None
if "books" not in st.session_state:
    st.session_state.books = []
if "background_jobs" not in st.session_state:
    st.session_state.background_jobs = {}
if "import_errors" not in st.session_state:
    st.session_state.import_errors = getattr(st.session_state, "import_errors", {})

# Background job runner helper
def run_background_job(job_id: str, fn, *args, **kwargs):
    def _target():
        st.session_state.background_jobs[job_id] = {"status": "running", "progress": 0, "error": None}
        try:
            for p in fn(*args, **kwargs):
                # fn yields progress dicts optionally
                if isinstance(p, dict):
                    st.session_state.background_jobs[job_id].update(p)
                time.sleep(0.01)
            st.session_state.background_jobs[job_id]["status"] = "completed"
            st.session_state.background_jobs[job_id]["progress"] = 100
        except Exception as e:
            st.session_state.background_jobs[job_id]["status"] = "error"
            st.session_state.background_jobs[job_id]["error"] = traceback.format_exc()
    t = threading.Thread(target=_target, daemon=True)
    t.start()
    return t

# Small no-op progress generator for compatibility if called functions are not generators
def _wrap_call_into_progress_generator(fn, *args, **kwargs):
    result = fn(*args, **kwargs)
    # If the function returned a generator already just yield from it
    if hasattr(result, "__iter__") and not isinstance(result, (str, bytes, dict)):
        for item in result:
            yield item
    else:
        # otherwise yield a few progress updates and finish
        yield {"progress": 20}
        time.sleep(0.2)
        yield {"progress": 60}
        time.sleep(0.2)
        yield {"progress": 100}

# Layout: sidebar for global controls, main area with tabs
st.set_page_config(page_title="Book Character Lab", layout="wide")

with st.sidebar:
    st.header("Book Character Lab")
    st.markdown("A cleaned, focused UI for ingesting books and chatting with characters.")

    # Show import errors if any
    if st.session_state.import_errors:
        with st.expander("Import errors (uploaded modules)", expanded=False):
            for k, v in st.session_state.import_errors.items():
                st.text_area(k, v, height=200)

    st.subheader("Upload / Select Book")
    uploaded_file = st.file_uploader("Upload a book (PDF / TXT / EPUB recommended)", type=["pdf", "txt", "epub"], help="Max recommended: 50 MB")
    if uploaded_file:
        save_dir = os.path.join("./books")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved to {save_path}")
        if save_path not in st.session_state.books:
            st.session_state.books.append(save_path)
            st.session_state.selected_book = save_path

    if st.session_state.books:
        st.selectbox("Choose a book", options=st.session_state.books, key="selected_book")

    st.markdown("---")
    st.subheader("Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Ingest Book"):
            if not st.session_state.selected_book:
                st.warning("Select or upload a book first.")
            elif ingest_mod is None:
                st.error("Ingest module not available. Please check uploads.")
            else:
                # run ingestion in background
                job_id = f"ingest_{os.path.basename(st.session_state.selected_book)}"
                run_background_job(job_id, lambda path=st.session_state.selected_book: _wrap_call_into_progress_generator(ingest_mod.ingest_new_book, path, False))
                st.success("Ingestion started (background). Check Jobs panel.")
    with col2:
        if st.button("Mine Knowledge"):
            if not st.session_state.selected_book:
                st.warning("Select or upload a book first.")
            elif kb_mod is None:
                st.error("Knowledge builder module not available. Please check uploads.")
            else:
                job_id = f"kb_{os.path.basename(st.session_state.selected_book)}"
                run_background_job(job_id, lambda path=st.session_state.selected_book: _wrap_call_into_progress_generator(kb_mod.run_knowledge_mining, path))
                st.success("Knowledge mining started (background). Check Jobs panel.")

    st.markdown("---")
    st.subheader("Settings")
    with st.expander("Model & Retrieval", expanded=False):
        st.text_input("LLM model (env)", value=os.environ.get("LLM_MODEL", "llama3.1"), key="ui_llm_model")
        st.number_input("Retrieval k", min_value=1, max_value=20, value=5, key="ui_retrieval_k")

    st.markdown("---")
    st.caption("Uploads saved to ./books. Imported modules expected at:")
    st.text(INGEST_PIPELINE_PATH)
    st.text(KNOWLEDGE_BUILDER_PATH)
    st.text(CHAT_ENGINE_PATH)

# Main area: header + tabs
st.title("Book Character Lab — Improved UI")

tabs = st.tabs(["Chat", "Characters", "Knowledge Graph", "Jobs & Logs"]) 

# Chat tab
with tabs[0]:
    st.subheader("Chat with book characters")
    colL, colR = st.columns([3,1])
    with colL:
        user_input = st.text_area("Your message", key="user_input", height=120)
        prompt_options = st.checkbox("Show advanced prompt options", value=False)
        if prompt_options:
            st.text_area("Prompt override", key="prompt_override", height=80)
        send = st.button("Send")
    with colR:
        st.markdown("**Active book**")
        st.write(st.session_state.selected_book or "— none —")
        st.markdown("**Retrieval k**")
        st.write(st.session_state.ui_retrieval_k)
        st.markdown("**Loaded modules**")
        st.write({
            "chat_engine": bool(chat_engine_mod),
            "ingest": bool(ingest_mod),
            "kb": bool(kb_mod)
        })

    if send:
        if not st.session_state.selected_book:
            st.warning("Select or upload a book first.")
        elif chat_engine_mod is None:
            st.error("Chat engine not available. Please upload chat_engine.py")
        else:
            try:
                # build a safe call path into the uploaded chat engine module
                chat_controller_cls = getattr(chat_engine_mod, "ChatController", None)
                if chat_controller_cls is None:
                    st.error("chat_engine.ChatController not found in uploaded module.")
                else:
                    controller = chat_controller_cls(book_path=st.session_state.selected_book)
                    # call controller in a try/except to avoid blocking the UI
                    res = controller.chat_once(user_input, k=st.session_state.ui_retrieval_k, prompt_override=st.session_state.get("prompt_override"))
                    st.session_state.messages.append(("user", user_input))
                    st.session_state.messages.append(("assistant", res))
            except Exception as e:
                st.error("Chat failed: " + str(e))

    # render chat history
    for who, content in st.session_state.messages[::-1]:
        if who == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:**")
            st.json(content)

# Characters tab
with tabs[1]:
    st.subheader("Detected characters & personas")
    char_col1, char_col2 = st.columns([2,3])
    with char_col1:
        st.write("Characters found (from ingestion)")
        chars_file = os.path.join("./books", os.path.basename(st.session_state.selected_book or ""), "characters.json")
        if os.path.exists(chars_file):
            try:
                import json
                with open(chars_file, "r") as fh:
                    chars = json.load(fh)
                st.write([c.get("name") for c in chars])
            except Exception:
                st.warning("characters.json found but failed to read.")
        else:
            st.info("No characters.json found. Run Ingest Book.")
    with char_col2:
        st.write("Edit persona templates (quick)")
        persona_name = st.text_input("Persona to edit (name)")
        persona_prompt = st.text_area("System prompt override", height=180)
        if st.button("Save persona locally"):
            if not persona_name:
                st.warning("Provide a persona name.")
            else:
                save_dir = os.path.join("./personas")
                os.makedirs(save_dir, exist_ok=True)
                pfile = os.path.join(save_dir, f"{persona_name}.json")
                import json
                with open(pfile, "w") as fh:
                    json.dump({"name": persona_name, "system_prompt": persona_prompt}, fh, indent=2)
                st.success(f"Saved persona to {pfile}")

# Knowledge Graph tab
with tabs[2]:
    st.subheader("Knowledge Graph & facts")
    kg_file = os.path.join("./books", os.path.basename(st.session_state.selected_book or ""), "knowledge.json")
    if os.path.exists(kg_file):
        try:
            import json
            with open(kg_file, "r") as fh:
                kg = json.load(fh)
            st.write(f"Triples: {len(kg.get('triples', []))}")
            if st.checkbox("Show triples"):
                st.json(kg.get("triples"))
        except Exception:
            st.warning("Failed to read knowledge.json")
    else:
        st.info("No knowledge.json found. Run Mine Knowledge.")

# Jobs & Logs tab
with tabs[3]:
    st.subheader("Background jobs & logs")
    for jid, info in st.session_state.background_jobs.items():
        st.write(jid)
        st.progress(int(info.get("progress", 0)))
        st.write(info.get("status"))
        if info.get("error"):
            with st.expander("Error"):
                st.code(info.get("error"))

    # Show a small logs area (from import errors or modules)
    if st.session_state.import_errors:
        st.subheader("Import Errors")
        for k, v in st.session_state.import_errors.items():
            st.text_area(k, v, height=200)

    st.markdown("---")
    st.caption("If you want this file to replace the existing app, save it to your project and run:\npython -m streamlit run streamlit_app_improved.py")

# End of improved app
