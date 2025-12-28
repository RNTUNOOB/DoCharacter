import os
import json
import logging
import asyncio
import time
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage

# Reuse the loader from your pipeline (or copy the function if you prefer decoupling)
from ingest_pipeline import load_document 

# --- CONFIGURATION ---
LLM_MODEL = "llama3.1"
LOG_FILE = "knowledge_builder.log"

# --- LOGGING ---
logger = logging.getLogger("KnowledgeBuilder")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

async def extract_facts_from_chunk(llm, chunk, chunk_id):
    """
    Asks Llama 3.1 to extract 3-5 high-value facts from a text segment.
    """
    prompt = """
    Extract 3 to 5 indisputable facts from this text.
    Focus on: Locations, Relationships, and Key Actions.
    
    Return a JSON object with a key "facts" containing a list of objects.
    Each fact object must have:
    - "subject": (e.g., "Alice")
    - "predicate": (e.g., "entered")
    - "object": (e.g., "The Rabbit Hole")
    """
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"TEXT:\n{chunk}")
        ])
        data = json.loads(response.content)
        facts = data.get("facts", [])
        # Add metadata
        for f in facts:
            f["source_chunk"] = chunk_id
        return facts
    except Exception as e:
        logger.warning(f"⚠️ Chunk {chunk_id} failed: {e}")
        return []

async def run_knowledge_mining(book_path):
    """
    Main entry point. Finds the original file, reads it, mines facts, saves JSON.
    """
    logger.info(f"⛏️ Starting Knowledge Mining for: {book_path}")
    
    # 1. Find the source file (PDF/EPUB) in the book directory
    files = [f for f in os.listdir(book_path) if f.lower().endswith(('.pdf', '.epub', '.txt'))]
    if not files:
        logger.error("❌ No source file found to mine.")
        return False
    
    source_file = os.path.join(book_path, files[0])
    text = load_document(source_file)
    
    # 2. Chunk the text (Large chunks for context)
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    logger.info(f"📘 Split text into {len(chunks)} segments for mining.")
    
    # 3. Initialize LLM
    llm = ChatOllama(model=LLM_MODEL, format="json", temperature=0.0, num_ctx=8192)
    
    # 4. Process Chunks (Sequential to save VRAM, or semi-parallel)
    all_facts = []
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        if i % 5 == 0:
            logger.info(f"   Mining segment {i+1}/{len(chunks)}...")
        
        facts = await extract_facts_from_chunk(llm, chunk, i)
        all_facts.extend(facts)
        
    # 5. Save to knowledge.json
    kb_path = os.path.join(book_path, "knowledge.json")
    
    # If file exists and isn't empty, maybe we merge? For now, we overwrite.
    with open(kb_path, "w", encoding='utf-8') as f:
        json.dump(all_facts, f, indent=4)
        
    logger.info(f"✅ Mining Complete. Extracted {len(all_facts)} facts in {time.time() - start_time:.2f}s.")
    return True

# Wrapper for Streamlit
def mine_knowledge_graph(book_path):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_knowledge_mining(book_path))

if __name__ == "__main__":
    # Test run via CLI
    import sys
    if len(sys.argv) > 1:
        mine_knowledge_graph(sys.argv[1])
    else:
        print("Usage: python knowledge_builder.py data/Book_Name")