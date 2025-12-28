import os
import shutil
import json
import logging
import asyncio
import time
import re
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

# --- 2. HIERARCHICAL STRUCTURE ANALYSIS (THE NEW BRAIN) ---

async def detect_structure(llm, text_preview):
    """
    Asks LLM to guess the best segmentation strategy based on the first 20k chars.
    It decides if the book has "Parts", "Chapters", or just "Scenes".
    """
    prompt = """
    Analyze the beginning of this book. Detect its structural hierarchy.
    Does it use "BOOK I", "PART 1", "CHAPTER 1", or just breaks?
    
    Return JSON:
    {
        "structure_type": "complex" OR "simple",
        "primary_separator": "Regex pattern to split by (e.g., 'CHAPTER \\d+') or 'none'",
        "estimated_depth": 1 or 2
    }
    """
    try:
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"TEXT START:\n{text_preview[:15000]}")
        ])
        config = json.loads(response.content)
        logger.info(f"🏗️  Detected Structure: {config}")
        return config
    except Exception as e:
        logger.warning(f"⚠️ Structure detection failed, defaulting to sliding window: {e}")
        return {"structure_type": "simple", "primary_separator": "none"}

def split_by_separator(text, separator):
    """Splits text based on LLM-suggested regex."""
    if separator == "none" or not separator:
        return [text]
    
    try:
        # Compile regex with ignore case
        pattern = re.compile(separator, re.IGNORECASE)
        parts = pattern.split(text)
        # Filter empty strings
        return [p for p in parts if len(p) > 500]
    except:
        return [text]

# --- 3. STANDARD AI WORKERS ---

async def analyze_timeline_chunk(llm, chunk, index, hierarchy_label=""):
    """Summarizes a chunk and extracts characters."""
    prompt = """
    Analyze this story segment.
    Return JSON:
    {
        "summary": "One sentence summary of events.",
        "characters_present": ["List of EXACT names"],
        "moods": [{"char": "Name", "emotion": "Emotion"}]
    }
    """
    try:
        res = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"CONTEXT: {hierarchy_label}\nTEXT:\n{chunk[:6000]}")
        ])
        data = json.loads(res.content)
        data['arc_id'] = index
        data['section'] = hierarchy_label # Store the section name (e.g., "Chapter 5")
        return data
    except Exception as e:
        logger.warning(f"⚠️ Timeline Chunk {index} failed: {e}")
        return None

async def generate_bio_from_history(llm, name, arc_summaries):
    """Creates a bio based on history."""
    context = "\n".join(arc_summaries)
    prompt = f"""
    Based on the following events, write a profile for "{name}".
    EVENTS:
    {context}
    
    Return JSON:
    {{
        "bio": "2 sentence bio.",
        "personality": "One word archetype.",
        "role": "Protagonist, Antagonist, or Support",
        "speaking_style": "Describe how they speak",
        "sample_quotes": []
    }}
    """
    try:
        res = await llm.ainvoke([SystemMessage(content=prompt)])
        return json.loads(res.content)
    except:
        return {"bio": "Unknown", "personality": "Unknown", "role": "Support", "speaking_style": "Normal", "sample_quotes": []}

def clean_name_for_dedup(name):
    return name.lower().replace("the ", "").strip()

async def build_hierarchical_timeline(text):
    """
    Smart Orchestrator:
    1. Detects Structure.
    2. Splits accordingly (by Chapter/Part if possible, else Sliding Window).
    3. Processes chunks.
    """
    logger.info("⏳ Starting Hierarchical Deep Scan...")
    
    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0.1, num_ctx=8192)
    
    # 1. Detect Structure
    structure = await detect_structure(llm, text)
    
    chunks = []
    labels = []
    
    # 2. Smart Splitting
    if structure.get("structure_type") == "complex" and structure.get("primary_separator") != "none":
        logger.info(f"✂️  Splitting by regex: {structure['primary_separator']}")
        raw_sections = split_by_separator(text, structure['primary_separator'])
        logger.info(f"   Found {len(raw_sections)} distinct sections.")
        
        # If sections are huge, sub-split them
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
        
        for i, section in enumerate(raw_sections):
            if len(section) < 500: continue # skip noise
            
            section_label = f"Section {i+1}"
            sub_chunks = text_splitter.split_text(section)
            for sub in sub_chunks:
                chunks.append(sub)
                labels.append(section_label)
                
    else:
        # Fallback to standard sliding window
        logger.info("✂️  Using Standard Sliding Window (8000 chars).")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=500)
        chunks = text_splitter.split_text(text)
        labels = [f"Part {i+1}" for i in range(len(chunks))]

    logger.info(f"📘 Processing {len(chunks)} total segments.")

    # 3. Process Timeline
    timeline = []
    char_appearances = [] 
    BANNED_NAMES = {"he", "she", "it", "they", "him", "her", "them", "who", "i", "you"}

    start_time = time.time()
    for i, chunk in enumerate(chunks):
        logger.info(f"   🔹 Analyzing {labels[i]} ({i+1}/{len(chunks)})...")
        # Pass the label (Chapter name) to the LLM for better context
        result = await analyze_timeline_chunk(llm, chunk, i+1, labels[i])
        if result:
            timeline.append(result)
            for name in result.get("characters_present", []):
                clean = name.strip()
                if len(clean) > 2 and clean.lower() not in BANNED_NAMES:
                    char_appearances.append((clean, result.get("summary", "")))
    
    logger.info(f"✅ Timeline built in {time.time() - start_time:.2f}s")

    # 4. Deduplication & Profiling (Same as before)
    logger.info("👥 Consolidating Cast List...")
    unique_names = list(set([c[0] for c in char_appearances]))
    unique_names.sort(key=len, reverse=True)
    
    canonical_map = {} 
    final_cast_events = {} 

    for name in unique_names:
        normalized = clean_name_for_dedup(name)
        match_found = False
        for canonical in final_cast_events.keys():
            canonical_norm = clean_name_for_dedup(canonical)
            if normalized in canonical_norm or canonical_norm in normalized:
                canonical_map[name] = canonical
                match_found = True
                break
        if not match_found:
            final_cast_events[name] = []
            canonical_map[name] = name

    for raw_name, summary in char_appearances:
        if raw_name in canonical_map:
            final_cast_events[canonical_map[raw_name]].append(summary)

    min_mentions = 2 if len(final_cast_events) > 10 else 1
    major_chars = {k: v for k, v in final_cast_events.items() if len(v) >= min_mentions}
    
    final_profiles = {}
    logger.info(f"✍️ Generating profiles for {len(major_chars)} characters...")
    
    for name, summaries in major_chars.items():
        logger.info(f"   Profiling: {name}")
        profile = await generate_bio_from_history(llm, name, summaries[:10]) 
        final_profiles[name] = profile

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
    
    # Calls the new Hierarchical Builder
    timeline_data, char_data = await build_hierarchical_timeline(text)
    
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