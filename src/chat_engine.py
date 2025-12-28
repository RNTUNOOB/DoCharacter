import os
import json
import logging
import time
import datetime
import gc
import networkx as nx
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# --- CONFIGURATION ---
LLM_MODEL = "llama3.1" 
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
LOG_FILE = "chat_logs.log"
CHAT_HISTORY_FILE = "conversation_history.jsonl"

# --- LOGGING SETUP ---
logger = logging.getLogger("ChatEngine")
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

def log_conversation_turn(book, char, user_query, bot_response, latency, context_len, arc_id=None):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "book": book,
        "character": char,
        "arc_id": arc_id,
        "latency_sec": round(latency, 2),
        "context_chars": context_len,
        "user_query": user_query,
        "bot_response": bot_response
    }
    try:
        with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write chat history: {e}")

class ChatController:
    def __init__(self):
        logger.info(f"⚡ Initializing Chat Engine with {LLM_MODEL}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True}
        )

        self.llm_creative = ChatOllama(
            model=LLM_MODEL,
            temperature=0.75,
            num_ctx=8192,
            format="json"
        )

        self.llm_strict = ChatOllama(
            model=LLM_MODEL,
            temperature=0.1,
            num_ctx=8192,
            format="json"
        )

        self.vector_store = None
        self.retriever = None
        self.knowledge_base = []
        self.current_book_path = None

    def load_book_resources(self, book_path):
        if self.current_book_path == book_path and self.vector_store:
            return

        start_ts = time.time()
        logger.info(f"📚 Loading resources from: {book_path}")
        db_path = os.path.join(book_path, "vector_db")
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector DB not found at {db_path}")

        try:
            self.vector_store = Chroma(
                persist_directory=db_path, 
                embedding_function=self.embeddings
            )
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10}
            )
            logger.info(f"✅ Resources loaded in {time.time() - start_ts:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Vector DB: {e}")
            self.vector_store = None

        kb_path = os.path.join(book_path, "knowledge.json")
        if os.path.exists(kb_path):
            try:
                with open(kb_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
            except:
                self.knowledge_base = []
        
        self.current_book_path = book_path

    def unload_resources(self):
        logger.info("🔓 Unloading resources and releasing DB locks...")
        self.vector_store = None
        self.retriever = None
        self.knowledge_base = []
        self.current_book_path = None
        gc.collect()

    def get_context(self, query):
        text_context = ""
        if self.retriever:
            docs = self.retriever.invoke(query)
            text_context = "\n---\n".join([d.page_content for d in docs])

        facts = []
        q_terms = query.lower().split()
        for entry in self.knowledge_base[:500]: 
            if any(term in str(entry.get('subject', '')).lower() for term in q_terms):
                facts.append(f"{entry.get('subject')} {entry.get('predicate')} {entry.get('object')}")
        
        fact_context = "\n".join(facts[:10])
        return text_context, fact_context

    def chat(self, user_query, char_name, char_data, current_arc_context, history=None, all_characters_data=None, full_timeline_data=None, selected_arc_id=None):
        turn_start = time.time()

        if not self.retriever:
            return {"response": "Please select a book first.", "tone": "System"}

        history = history or []
        context_text, context_facts = self.get_context(user_query)
        
        # --- FILTER TIMELINE ---
        past_events = []
        if full_timeline_data and selected_arc_id:
            valid_events = [arc for arc in full_timeline_data if arc.get('arc_id', 999) <= selected_arc_id]
            # Include Section Headers in summary for better context
            past_events = [f"[{e.get('section', 'General')}] {e.get('summary')}" for e in valid_events]
        
        timeline_context = "\n".join(past_events)

        # --- 1. AUDITOR MODE ---
        if char_name == "The_Auditor" or char_data.get("role") == "auditor":
            result = self._run_auditor(user_query, context_text, context_facts, timeline_context)
        
        # --- 2. CHARACTER MODE ---
        else:
            result = self._run_character(
                user_query, char_name, char_data, 
                context_text, current_arc_context, history
            )

        latency = time.time() - turn_start
        response_text = result.get("response") or result.get("answer") or "..."
        book_name = os.path.basename(self.current_book_path) if self.current_book_path else "Unknown"

        logger.info(f"🗣️ {char_name} replied in {latency:.2f}s (Context: {len(context_text)} chars)")
        
        log_conversation_turn(
            book=book_name,
            char=char_name,
            user_query=user_query,
            bot_response=response_text,
            latency=latency,
            context_len=len(context_text),
            arc_id=selected_arc_id
        )

        return result

    def _run_auditor(self, query, text, facts, timeline_history):
        logger.info("🕵️ Running Auditor Logic with Timeline Filter...")
        
        system_prompt = """You are The Auditor, an objective AI fact-checker.
        You observe the story as it unfolds.
        
        CRITICAL RULES:
        1. You only know what has happened in the "KNOWN HISTORY" provided below.
        2. Do NOT use outside knowledge of the book. If an event is not in the history or context, it hasn't happened yet.
        3. Answer factually.
        
        Output format (JSON):
        {
            "answer": "The factual answer...",
            "confidence": 0.0 to 1.0,
            "sources": ["List of specific quotes used"]
        }
        """

        user_input = f"""
        KNOWN HISTORY (Chronological Order):
        {timeline_history}

        RELEVANT TEXT EXCERPTS:
        {text}
        
        RELEVANT FACTS:
        {facts}
        
        USER QUESTION: {query}
        """

        try:
            response = self.llm_strict.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
            )
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Auditor Error: {e}")
            return {"answer": "Error analyzing data.", "confidence": 0.0, "sources": []}

    def _run_character(self, query, name, data, text_context, arc_context, history):
        logger.info(f"🎭 Running Character Logic for {name}...")

        bio = data.get('bio', 'Unknown')
        personality = data.get('personality', 'Neutral')
        style = data.get('speaking_style', 'Standard')
        quotes = data.get('sample_quotes', [])
        
        chat_log = []
        for msg in history[-5:]:
            role = msg.get('role')
            content = msg.get('content')
            if isinstance(content, dict): 
                content = content.get('response', '')
            chat_log.append(f"{role.upper()}: {content}")
        history_str = "\n".join(chat_log)

        system_prompt = f"""You are {name}.
        BIO: {bio}
        PERSONALITY: {personality}
        SPEAKING STYLE: {style}
        SAMPLE QUOTES: {json.dumps(quotes)}
        
        CURRENT SCENE MOOD: {arc_context}
        
        INSTRUCTIONS:
        1. Read the Context and User Query.
        2. Formulate an internal thought.
        3. Reply to the user in character. 
        4. STAY IN CHARACTER.
        
        Output format (JSON):
        {{
            "internal_thought": {{
                "analysis": "...",
                "strategy": "..."
            }},
            "response": "Your spoken reply...",
            "tone": "Emotion",
            "sources": []
        }}
        """

        user_input = f"""
        BOOK_CONTEXT: {text_context}
        CONVERSATION_HISTORY:
        {history_str}
        
        USER_SAYS: {query}
        """

        try:
            response = self.llm_creative.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
            )
            return json.loads(response.content)
        except Exception as e:
            logger.error(f"Character Error: {e}")
            return {
                "response": "...", 
                "tone": "Confused", 
                "internal_thought": {"analysis": "Error", "strategy": "Fallback"}
            }