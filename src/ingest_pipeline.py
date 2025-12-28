import os
import shutil
import json
import logging
import asyncio
import time
from collections import Counter
from bs4 import BeautifulSoup
from pypdf import PdfReader
import ebooklib
from ebooklib import epub

# LangChain Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

# --- CONFIGURATION ---
LLM_MODEL = "llama3.1"
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
INGEST_LOG_FILE = "ingest_logs.log"

# --- LOGGING SETUP ---
logger = logging.getLogger("IngestPipeline")
logger.setLevel(logging.INFO)
# Clear existing handlers to prevent duplicates on reload
if logger.hasHandlers():
    logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
fh = logging.FileHandler(INGEST_LOG_FILE, encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# --- 1. DOCUMENT LOADING ---
def load_document(path):
    """Reads PDF, EPUB, or TXT and returns raw text."""
    start_ts = time.time()
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    logger.info(f"📂 Loading file: {path}")
    
    text = ""
    try:
        if ext == ".pdf":
            reader = PdfReader(path)
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"
        elif ext == ".epub":
            book = epub.read_epub(path)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
        else:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
    except Exception as e:
        logger.error(f"❌ Failed to load document: {e}")
        raise

    clean_text = " ".join(text.split())
    logger.info(f"✅ Loaded {len(clean_text)} characters in {time.time() - start_ts:.2f}s")
    return clean_text

# --- 2. AI WORKERS ---

async def analyze_timeline_chunk(llm, chunk, index):
    """Summarizes a chunk and extracts characters present in that chunk."""
    prompt = """
    Analyze this story segment.
    Return JSON:
    {
        "summary": "One sentence summary of events.",
        "characters_present": ["List of EXACT names of characters appearing here"],
        "moods": [{"char": "Name", "emotion": "Emotion"}]
    }
    """
    try:
        res = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"TEXT:\n{chunk[:6000]}") # limit chunk context for speed
        ])
        data = json.loads(res.content)
        data['arc_id'] = index
        return data
    except Exception as e:
        logger.warning(f"⚠️ Timeline Chunk {index} failed: {e}")
        return None

async def generate_bio_from_history(llm, name, arc_summaries):
    """Creates a bio based on what the character actually DID in the timeline."""
    context = "\n".join(arc_summaries)
    prompt = f"""
    Based on the following story events, write a character profile for "{name}".
    EVENTS:
    {context}
    
    Return JSON:
    {{
        "bio": "2 sentence bio based on these events.",
        "personality": "One word archetype.",
        "role": "Protagonist, Antagonist, or Support",
        "speaking_style": "Describe how they speak (e.g. Formal, Rude)",
        "sample_quotes": []
    }}
    """
    try:
        res = await llm.ainvoke([SystemMessage(content=prompt)])
        return json.loads(res.content)
    except:
        return {
            "bio": "A mysterious figure mentioned in the story.", 
            "personality": "Unknown", 
            "role": "Support",
            "speaking_style": "Normal",
            "sample_quotes": []
        }

def clean_name_for_dedup(name):
    """Standardizes names for comparison (e.g. 'The White Rabbit' -> 'white rabbit')"""
    return name.lower().replace("the ", "").strip()

async def build_timeline_and_extract_cast(text):
    """
    1. Builds timeline from WHOLE book.
    2. Aggregates characters found in timeline.
    3. SMART MERGE: Deduplicates names (White Rabbit == The White Rabbit).
    4. Generates profiles for top characters.
    """
    logger.info("⏳ Starting Deep Scan (Timeline + Cast Extraction)...")
    
    # 1. Split Book
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    logger.info(f"📘 Split book into {len(chunks)} segments.")
    
    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0.1, num_ctx=8192)
    
    # 2. Build Timeline
    timeline = []
    char_appearances = [] # List of (Name, Arc_Summary) tuples
    
    # Banned words to prevent pronouns from becoming characters
    BANNED_NAMES = {"he", "she", "it", "they", "him", "her", "them", "who", "i", "you"}

    start_time = time.time()
    for i, chunk in enumerate(chunks):
        logger.info(f"   🔹 Analyzing Segment {i+1}/{len(chunks)}...")
        result = await analyze_timeline_chunk(llm, chunk, i+1)
        if result:
            timeline.append(result)
            for name in result.get("characters_present", []):
                clean = name.strip()
                # FIX: Removed the (" " in clean) check that killed Alice
                if len(clean) > 2 and clean.lower() not in BANNED_NAMES:
                    char_appearances.append((clean, result.get("summary", "")))
    
    logger.info(f"✅ Timeline built in {time.time() - start_time:.2f}s")

    # 3. Smart Deduplication
    logger.info("👥 Consolidating & Merging Cast List...")
    
    # Sort by length desc so we match "Queen of Hearts" before "Queen"
    unique_names = list(set([c[0] for c in char_appearances]))
    unique_names.sort(key=len, reverse=True)
    
    canonical_map = {} # {'The Queen': 'Queen of Hearts', 'Queen': 'Queen of Hearts'}
    final_cast_events = {} # {'Queen of Hearts': [summaries...]}

    for name in unique_names:
        normalized = clean_name_for_dedup(name)
        match_found = False
        
        # Check against existing canonical names
        for canonical in final_cast_events.keys():
            canonical_norm = clean_name_for_dedup(canonical)
            # Logic: If "Queen" is in "Queen of Hearts" OR "White Rabbit" == "white rabbit"
            if normalized in canonical_norm or canonical_norm in normalized:
                canonical_map[name] = canonical
                match_found = True
                break
        
        if not match_found:
            final_cast_events[name] = []
            canonical_map[name] = name

    # Aggregate stories into the canonical buckets
    for raw_name, summary in char_appearances:
        if raw_name in canonical_map:
            root_name = canonical_map[raw_name]
            final_cast_events[root_name].append(summary)

    # Filter for characters mentioned in at least 2 chunks (noise filter)
    # Exception: If the cast is very small, allow everyone.
    min_mentions = 2 if len(final_cast_events) > 10 else 1
    major_chars = {k: v for k, v in final_cast_events.items() if len(v) >= min_mentions}
    
    # 4. Generate Profiles
    final_profiles = {}
    logger.info(f"✍️ Generating profiles for {len(major_chars)} unique characters...")
    
    for name, summaries in major_chars.items():
        logger.info(f"   Profiling: {name}")
        profile = await generate_bio_from_history(llm, name, summaries[:10]) 
        final_profiles[name] = profile

    # Add Auditor
    final_profiles["The_Auditor"] = {
        "role": "auditor",
        "bio": "An impartial AI fact-checker.",
        "personality": "Objective",
        "speaking_style": "Analytical",
        "sample_quotes": []
    }

    return timeline, final_profiles

# --- 3. VECTORIZATION ---
def create_vector_db(text, save_path):
    start_ts = time.time()
    logger.info("🧠 Vectorizing text (CPU)...")
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    docs = splitter.create_documents([text])
    
    # Batch processing
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        if i == 0:
            Chroma.from_documents(batch, embeddings, persist_directory=save_path)
        else:
            db = Chroma(persist_directory=save_path, embedding_function=embeddings)
            db.add_documents(batch)
        if i % 500 == 0:
            logger.info(f"   Encoded {i}/{len(docs)} chunks...")
            
    logger.info(f"✅ Vector DB finished in {time.time() - start_ts:.2f}s")

# --- 4. ORCHESTRATOR ---
async def run_pipeline(file_obj, book_title):
    overall_start = time.time()
    clean_title = "".join(x for x in book_title if x.isalnum() or x in " _-").strip().replace(" ", "_")
    base_dir = os.path.join("data", clean_title)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    file_path = os.path.join(base_dir, file_obj.name)
    with open(file_path, "wb") as f:
        f.write(file_obj.getbuffer())
        
    text = load_document(file_path)
    
    # Unified Step: Build Timeline AND Extract Characters from it
    timeline_data, char_data = await build_timeline_and_extract_cast(text)
    
    with open(os.path.join(base_dir, "timeline.json"), "w") as f:
        json.dump(timeline_data, f, indent=4)
    with open(os.path.join(base_dir, "characters.json"), "w") as f:
        json.dump(char_data, f, indent=4)
    with open(os.path.join(base_dir, "knowledge.json"), "w") as f:
        json.dump([], f)
        
    await asyncio.to_thread(create_vector_db, text, os.path.join(base_dir, "vector_db"))
    
    logger.info(f"🎉 Ingestion Complete. Total time: {time.time() - overall_start:.2f}s")
    return clean_title

def ingest_new_book(file_obj, book_title):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_pipeline(file_obj, book_title))