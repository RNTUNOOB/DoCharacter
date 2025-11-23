import os
import logging
import json
import networkx as nx
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL = "llama3.1"
LOG_FILE = "chat_logs.log"

# --- LOGGING SETUP ---
logger = logging.getLogger("ChatEngine")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(stream_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
# ---------------------

class ChatController:
    def __init__(self):
        print("⚡ Initializing AI Models...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_ID,
            model_kwargs={'device': 'cpu', 'trust_remote_code': True}
        )
        self.llm = OllamaLLM(model=LLM_MODEL, temperature=0.7)
        self.fact_llm = OllamaLLM(model=LLM_MODEL, temperature=0.1)
        
        self.vector_store = None
        self.retriever = None
        self.graph = None 
        self.current_book_path = None
        self.knowledge_base = []

    def load_book_resources(self, book_path):
        if self.current_book_path == book_path and self.vector_store is not None:
            return

        db_path = os.path.join(book_path, "vector_db")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector DB missing at {db_path}")

        print(f"📚 Switching Context to: {book_path}")
        self.vector_store = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}) 
        
        graph_path = os.path.join(book_path, "relationships.graphml")
        if os.path.exists(graph_path):
            try:
                self.graph = nx.read_graphml(graph_path)
                print("🕸️ Social Graph Loaded.")
            except: self.graph = None
        else: self.graph = None

        # NEW: Load Knowledge Base (GraphRAG Facts)
        kb_path = os.path.join(book_path, "knowledge.json")
        if os.path.exists(kb_path):
            with open(kb_path, 'r') as f:
                self.knowledge_base = json.load(f)
        else:
            self.knowledge_base = []

        self.current_book_path = book_path

    def get_social_context(self, char_name):
        if not self.graph or not self.graph.has_node(char_name): return "No known relationships."
        rels = []
        for neighbor in self.graph.neighbors(char_name):
            edge = self.graph.get_edge_data(char_name, neighbor)
            rels.append(f"- {edge.get('relation', 'knows')} {neighbor}")
        return "\n".join(rels) if rels else "No close contacts."

    def get_relevant_context(self, query):
        if not self.retriever: return "No book selected."
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    # --- STRATEGY 1: DOSSIER INJECTION ---
    def get_referenced_char_profiles(self, query, all_characters_data):
        """If the user mentions 'O'Brien', fetch O'Brien's bio."""
        profiles = []
        query_lower = query.lower()
        
        for name, data in all_characters_data.items():
            if name.lower() in query_lower or query_lower in name.lower():
                profile = f"--- CHARACTER DOSSIER: {name} ---\nBio: {data.get('bio')}\nSecrets: {data.get('secrets')}\n"
                profiles.append(profile)
        
        return "\n".join(profiles)

    # --- STRATEGY 2: SYNOPSIS INJECTION ---
    def get_full_timeline_summary(self, full_timeline):
        """Condenses the entire book timeline into a short summary list."""
        if not full_timeline: return "No timeline available."
        summary = []
        for event in full_timeline:
            summary.append(f"Arc {event.get('arc_id')}: {event.get('summary')}")
        return "\n".join(summary)

    # --- STRATEGY 3: SEMANTIC FACT LOOKUP (GraphRAG) ---
    def get_relevant_facts(self, query):
        """Simple keyword/semantic filter for the knowledge base."""
        query_words = set(query.lower().split())
        hits = []
        for fact in self.knowledge_base:
            # Check matches in Subject or Object
            s, o = fact.get('subject', '').lower(), fact.get('object', '').lower()
            if any(w in s for w in query_words) or any(w in o for w in query_words):
                hits.append(f"- {fact['subject']} {fact['predicate']} {fact['object']} ({fact['context']})")
        return "\n".join(hits[:12]) # Top 12 relevant facts

    def chat(self, user_query, char_name, char_data, current_arc_context, 
             history=[], all_characters_data={}, full_timeline_data=[]): 
        
        if not self.retriever: return "Please select a book first."

        # 1. Retrieval (Vector + Graph)
        text_context = self.get_relevant_context(user_query)
        fact_context = self.get_relevant_facts(user_query) # <--- NEW
        
        # 2. Strategy 1: Referenced characters
        relevant_dossiers = self.get_referenced_char_profiles(user_query, all_characters_data)
        
        # 3. Strategy 2: Full Timeline
        full_book_summary = self.get_full_timeline_summary(full_timeline_data)

        # 4. Format History
        formatted_history = "\n".join([f"{m['role'].title()}: {m['content']}" for m in history[-5:]])

        logger.info(f"--- TURN: {char_name} | Query: {user_query} ---")

        # --- MODE A: AUDITOR ---
        if char_name == "The_Auditor":
            prompt = PromptTemplate.from_template("""
            You are 'The Auditor', an external AI analyst reviewing a book.
            
            VERIFIED FACTS (Knowledge Graph):
            {facts}
            
            GLOBAL PLOT SUMMARY (Timeline):
            {full_book_summary}
            
            RELEVANT CHARACTER DOSSIERS:
            {dossiers}
            
            SPECIFIC TEXT EXCERPTS (Narrative):
            {context}
            
            USER QUESTION: {query}
            
            INSTRUCTIONS:
            1. Combine the verified facts, plot summary, and text excerpts to answer.
            2. If the user asks about a character's fate (e.g., "What happened to X?"), prioritize the Facts and Dossiers.
            3. Be factual and objective.
            
            ANSWER:
            """)
            chain = prompt | self.fact_llm
            final_response = chain.invoke({
                "context": text_context,
                "facts": fact_context, 
                "full_book_summary": full_book_summary,
                "dossiers": relevant_dossiers,
                "query": user_query
            })

        # --- MODE B: PERSONA ---
        else:
            social_context = self.get_social_context(char_name)

            # Inner Monologue
            thought_prompt = PromptTemplate.from_template("""
            You are {name}. Bio: {bio}. Personality: {personality}.
            
            SITUATION:
            - Talking to a Stranger (User).
            - User said: "{query}"
            
            YOUR KNOWLEDGE:
            - Verified Facts: {facts}
            - Narrative Memory: {context}
            - Relevant Dossiers: {dossiers}
            
            TASK: Think silently about the user's input. 
            Do NOT confuse the User with characters from your memories.
            Output ONLY the thought.
            """)
            
            thought_chain = thought_prompt | self.llm
            inner_monologue = thought_chain.invoke({
                "name": char_name,
                "bio": char_data['bio'],
                "personality": char_data['personality'],
                "facts": fact_context,
                "dossiers": relevant_dossiers,
                "query": user_query,
                "context": text_context
            })
            log_thought = inner_monologue

            # External Speech
            samples = "\n".join([f'- "{s}"' for s in char_data.get('sample_quotes', [])])
            
            speech_prompt = PromptTemplate.from_template("""
            You are {name}. Voice Style: {style}.
            
            VOICE SAMPLES:
            {samples}
            
            CHAT HISTORY:
            {chat_history}
            
            CONTEXT:
            User: "{query}"
            Your Thought: {thought}
            Memory: {context}
            
            Respond naturally.
            """)
            
            speech_chain = speech_prompt | self.llm
            final_response = speech_chain.invoke({
                "name": char_name,
                "style": char_data['speaking_style'],
                "samples": samples,
                "chat_history": formatted_history,
                "context": text_context,
                "query": user_query,
                "thought": inner_monologue
            })

        logger.info(f"🧠 THOUGHT: {log_thought}")
        logger.info(f"🗣️ RESPONSE: {final_response}")
        return final_response