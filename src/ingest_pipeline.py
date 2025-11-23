import os
import shutil
import json
import re
import math
import spacy
import asyncio
import networkx as nx
from collections import Counter
from bs4 import BeautifulSoup
from pypdf import PdfReader
import ebooklib
from ebooklib import epub

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# "The Intern": Fast model for reading/summarizing (3B params)
FAST_MODEL = "llama3.2" 

# "The Embedding Engine": CPU-based
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"

# Load Spacy
try:
    nlp = spacy.load("en_core_web_lg")
except:
    print("⚠️ 'en_core_web_lg' not found. Falling back to 'sm'.")
    nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000 

# --- HELPER: ROBUST JSON EXTRACTION ---
def extract_json(text):
    text = text.strip()
    text = re.sub(r'^```json', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```', '', text, flags=re.MULTILINE)
    start_idx = text.find('{')
    if start_idx == -1: return None
    try:
        # Find the last closing brace to avoid trailing garbage
        end_idx = text.rfind('}')
        if end_idx == -1: return None
        return json.loads(text[start_idx:end_idx+1].replace('\n', ' '))
    except: return None

# --- 1. LOADERS ---
def load_document(path):
    if not os.path.exists(path): raise FileNotFoundError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        text = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n".join(text)
    elif ext == ".epub":
        book = epub.read_epub(path)
        text = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text.append(soup.get_text())
        return "\n".join(text)
    elif ext == ".txt":
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    else: raise ValueError(f"Unsupported format: {ext}")

# --- 2. FAST CHARACTER SCAN (CPU) ---
def scan_characters(text):
    print(f"🔍 Deep Scanning for Characters ({len(text)} chars)...")
    
    WINDOW_SIZE = 100000 
    OVERLAP = 20000      
    global_counts = Counter()
    cursor = 0
    
    stop_list = {
        "the", "one", "chapter", "mr", "mrs", "miss", "dr", "sir", "madam", 
        "comrade", "brother", "ii", "iii", "master", "mistress", "st", 
        "page", "vol", "volume", "author", "illustration", "copyright", "part",
        "book", "prologue", "epilogue", "contents", "project", "gutenberg"
    }
    
    bad_suffixes = {
        "said", "asked", "replied", "cried", "shouted", "whispered", "muttered",
        "thought", "felt", "looked", "turned", "went", "came", "saw", "walked",
        "desperately", "suddenly", "quietly", "angrily", "happily", "gently",
        "quickly", "desper", "began", "continued"
    }

    while cursor < len(text):
        end = min(cursor + WINDOW_SIZE, len(text))
        chunk = text[cursor:end]
        doc = nlp(chunk)
        
        valid_names = []
        for ent in doc.ents:
            # FIX 1: Enhanced Check for Irish Names (O'Brien / O’Brien)
            # We explicitly check for both straight quote (') and curly quote (’)
            is_person = ent.label_ == "PERSON"
            name_raw = ent.text.strip()
            is_irish = (ent.label_ == "ORG" or ent.label_ == "GPE") and (name_raw.startswith("O'") or name_raw.startswith("O’"))
            
            if is_person or is_irish:
                name = name_raw.replace("\n", " ")
                
                # A. Punctuation Splitter
                for char in ["—", "–", "‘", "“", "("]: 
                    if char in name:
                        name = name.split(char)[0].strip()

                # B. Possessive Removal
                name = re.sub(r"['’]s$", "", name)

                # C. Bad Suffix Cleaner
                parts = name.split()
                if len(parts) > 1:
                    last_word = parts[-1].lower()
                    last_word_clean = re.sub(r'[^\w\s]', '', last_word)
                    
                    if last_word_clean in bad_suffixes:
                        name = " ".join(parts[:-1])
                    elif last_word_clean.endswith("ly") and len(last_word_clean) > 3:
                         name = " ".join(parts[:-1])

                clean_name_key = name.lower().replace(".", "").strip()
                if len(name) > 2 and clean_name_key not in stop_list:
                    # FIX 2: Ensure the regex permits both straight and curly apostrophes for O'Names
                    name = re.sub(r"[^\w\s'\-’]", "", name)
                    valid_names.append(name.strip())
                    
        global_counts.update(valid_names)
        cursor += (WINDOW_SIZE - OVERLAP)
        if end == len(text): break
            
    final_candidates = {}
    
    # D. Case-Insensitive Merging
    for name, count in global_counts.most_common(300): 
        clean_name = name.lower().replace(".", "")
        if len(name) < 3 or clean_name in stop_list: continue
        
        merged = False
        for existing in final_candidates:
            if name.lower() in existing.lower() or existing.lower() in name.lower():
                if len(name) > len(existing):
                    final_candidates[name] = final_candidates.pop(existing) + count
                else:
                    final_candidates[existing] += count
                merged = True
                break
        if not merged:
            final_candidates[name] = count
            
    top_chars = sorted(final_candidates, key=final_candidates.get, reverse=True)[:20]
    print(f"🧠 Identified Cast: {top_chars}")
    return top_chars

# --- 3. KNOWLEDGE EXTRACTION (NEW - GraphRAG) ---
class KnowledgeTriplet(BaseModel):
    subject: str = Field(description="The main entity (Character/Place)")
    predicate: str = Field(description="The action or relationship (e.g., 'killed', 'loves', 'is inside')")
    object: str = Field(description="The target entity")
    context: str = Field(description="A brief 5-word context (e.g., 'during the hate week')")

class KnowledgeChunk(BaseModel):
    facts: list[KnowledgeTriplet]

async def extract_facts(llm, parser, chunk):
    prompt = PromptTemplate(
        template="""Extract 5 key facts from text as Subject-Action-Object.
        Focus on: Major plot events, character relationships, secrets.
        TEXT: {text}
        {format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    try:
        res = await (prompt | llm).ainvoke({"text": chunk})
        data = extract_json(res)
        return data.get("facts", []) if data else []
    except: return []

async def build_knowledge_base(text, output_path):
    print("🧠 Mining Knowledge Base (Llama 3.2)...")
    # OPTIMIZATION: Increased context to 8k to use available VRAM
    llm = OllamaLLM(model=FAST_MODEL, temperature=0.0, num_ctx=8192)
    parser = PydanticOutputParser(pydantic_object=KnowledgeChunk)
    
    chunk_size = 8000
    tasks = []
    # Process text in chunks
    for i in range(0, len(text), chunk_size):
        tasks.append(extract_facts(llm, parser, text[i:i+chunk_size]))
    
    results = await asyncio.gather(*tasks)
    all_facts = [f for sub in results for f in sub]
    
    with open(output_path, 'w') as f:
        json.dump(all_facts, f, indent=4)
    print(f"✅ Extracted {len(all_facts)} Facts.")

# --- 4. ASYNC TASKS (EXISTING) ---

# A. PROFILE GENERATION
class CharacterProfile(BaseModel):
    name: str = Field(description="Name of the character")
    bio: str = Field(description="Short biography (1 sentence)")
    personality: str = Field(description="Key personality traits")
    speaking_style: str = Field(description="How they speak (keywords)")
    sample_quotes: list[str] = Field(description="3 short quotes SPOKEN BY this character.")

async def generate_single_profile(llm, parser, char, text):
    # OPTIMIZATION: Reduced context window for faster processing
    snippets = []
    for p in text.split('\n\n'):
        if char in p and len(p) > 50:
            snippets.append(p.replace("\n", " ").strip())
            if len(snippets) >= 8: break  # Reduced from 15 to 8 for speed
    context = "\n---\n".join(snippets)
    if not context: return None

    prompt = PromptTemplate(
        template="""Profile "{target_name}".
        Context: {context}
        {format_instructions}
        """,
        input_variables=["target_name", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm
    try:
        res = await chain.ainvoke({"target_name": char, "context": context})
        data = extract_json(res)
        if data and (char.lower() in data.get("name", "").lower() or data.get("name", "").lower() in char.lower()):
            print(f"   ✅ Profiled: {char}")
            return char, data
    except Exception as e:
        print(f"   ❌ Failed Profile {char}: {e}")
    return None

async def run_async_profiler(text, top_chars, output_path):
    # OPTIMIZATION: Increased context to 8k
    llm = OllamaLLM(model=FAST_MODEL, temperature=0.1, num_ctx=8192)
    parser = PydanticOutputParser(pydantic_object=CharacterProfile)
    
    # Profile all 20 characters in parallel
    tasks = [generate_single_profile(llm, parser, char, text) for char in top_chars]
    results = await asyncio.gather(*tasks)
    
    profiles = {}
    profiles["The_Auditor"] = {
        "name": "The_Auditor", "bio": "Fact checker AI", "personality": "Objective", 
        "speaking_style": "Analytical", "sample_quotes": [], "mental_state": "Stable", "secrets": "None"
    }
    
    for res in results:
        if res: profiles[res[0]] = res[1]

    with open(output_path, 'w') as f:
        json.dump(profiles, f, indent=4)

# B. TIMELINE GENERATION
class TimelineEvent(BaseModel):
    arc_id: int
    summary: str
    moods: dict[str, str] = Field(description="Mood of main characters in this scene")
    characters_present: list[str]

async def analyze_single_arc(llm, parser, arc_id, chunk):
    prompt = PromptTemplate(
        template="""Analyze Arc {arc_id}.
        TEXT: {text}
        
        TASK:
        1. Summarize event (1 sentence).
        2. List characters present.
        3. Determine moods.
        
        {format_instructions}
        """,
        input_variables=["arc_id", "text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = prompt | llm
    try:
        # Retry logic built-in
        for _ in range(2):
            res = await chain.ainvoke({"text": chunk, "arc_id": arc_id})
            data = extract_json(res)
            if data: 
                print(f"   ✅ Analyzed Arc {arc_id}")
                return data
    except Exception as e:
        print(f"   ⚠️ Failed Arc {arc_id}: {e}")
    return None

async def run_async_timeline(text, output_path):
    # OPTIMIZATION: Increased context to 8k
    llm = OllamaLLM(model=FAST_MODEL, temperature=0.1, num_ctx=8192)
    parser = PydanticOutputParser(pydantic_object=TimelineEvent)
    
    # OPTIMIZATION: We can now chunk bigger!
    CHUNK_SIZE = 35000  # Increased from 25k since we have 8k context
    OVERLAP = 2000
    
    tasks = []
    cursor = 0
    arc_id = 1
    
    while cursor < len(text):
        end = min(cursor + CHUNK_SIZE, len(text))
        chunk = text[cursor:end]
        
        # Pass larger chunk to LLM (6k chars fits easily in 8k tokens)
        analysis_chunk = chunk[:6000] 
        
        tasks.append(analyze_single_arc(llm, parser, arc_id, analysis_chunk))
        
        cursor += (CHUNK_SIZE - OVERLAP)
        arc_id += 1
        if end == len(text): break
    
    results = await asyncio.gather(*tasks)
    valid_results = [r for r in results if r]
    valid_results.sort(key=lambda x: x['arc_id'])

    with open(output_path, 'w') as f:
        json.dump(valid_results, f, indent=4)

# --- 5. GRAPH BUILDER (OPTIMIZED) ---
def run_graph_builder(text, characters, output_path):
    print("🕸️ Building Social Graph (Fast Mode)...")
    G = nx.Graph()
    for char in characters: G.add_node(char)
    
    # OPTIMIZATION: Increased context to 8k
    llm = OllamaLLM(model=FAST_MODEL, temperature=0.1, num_ctx=8192)
    
    # OPTIMIZATION: Analyze slightly more scenes
    interaction_scenes = []
    for p in text.split('\n\n'):
        found = [c for c in characters if c in p]
        if len(found) >= 2 and len(p) > 50: 
            interaction_scenes.append(p.replace("\n", " ").strip())
    
    target_scenes = interaction_scenes[:12] # Increased from 8 to 12
    
    if not target_scenes:
        nx.write_graphml(G, output_path)
        return

    prompt = PromptTemplate.from_template("""
    Identify relationships: {characters}.
    TEXT: {text}
    Format: Source | Relation | Target
    Output ONLY lines.
    """)
    chain = prompt | llm
    try:
        res = chain.invoke({"characters": ", ".join(characters), "text": "\n".join(target_scenes)})
        for line in res.split('\n'):
            if "|" in line:
                parts = line.split("|")
                if len(parts) == 3:
                    G.add_edge(parts[0].strip(), parts[2].strip(), relation=parts[1].strip())
    except: pass
    nx.write_graphml(G, output_path)

# --- 6. VECTORIZER (CPU) ---
def run_vectorizer(text, db_path, source_name):
    print(f"🧠 Initializing CPU Embeddings ({EMBED_MODEL_ID})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_ID,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True}
    )
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    docs = splitter.create_documents([text])
    for i, doc in enumerate(docs):
        doc.metadata = {"source": source_name, "chunk_id": i}

    if os.path.exists(db_path): shutil.rmtree(db_path)
    print(f"💾 Vectorizing {len(docs)} chunks (CPU)...")
    
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        Chroma.from_documents(documents=docs[i:i+batch_size], embedding=embeddings, persist_directory=db_path)
    print("✅ Memory Ready.")

# --- MASTER FUNCTION ---
async def pipeline_orchestrator(text, base_dir, file_name):
    top_chars = scan_characters(text)
    
    print("🚀 Starting Fully Parallel Processing (Vectorization + LLM)...")

    # Task A: Vectorization (CPU Bound)
    vector_task = asyncio.to_thread(
        run_vectorizer, 
        text, 
        os.path.join(base_dir, "vector_db"), 
        file_name
    )

    # Task B: Graph Builder (LLM Fast Mode)
    graph_task = asyncio.to_thread(
        run_graph_builder, 
        text, 
        top_chars, 
        os.path.join(base_dir, "relationships.graphml")
    )

    # Task C: LLM Analysis (Async Fast Mode)
    profiler_task = run_async_profiler(
        text, 
        top_chars, 
        os.path.join(base_dir, "characters.json")
    )
    
    timeline_task = run_async_timeline(
        text, 
        os.path.join(base_dir, "timeline.json")
    )

    # NEW TASK: Knowledge Base Extraction
    kb_task = build_knowledge_base(
        text, 
        os.path.join(base_dir, "knowledge.json")
    )

    await asyncio.gather(
        vector_task, 
        graph_task, 
        profiler_task, 
        timeline_task,
        kb_task
    )
    
    print("✅ Ingestion Pipeline Complete.")

def ingest_new_book(file_obj, book_title):
    clean_title = "".join(x for x in book_title if x.isalnum() or x in " _-").strip().replace(" ", "_")
    base_dir = os.path.join("data", clean_title)
    if not os.path.exists(base_dir): os.makedirs(base_dir)
    
    file_path = os.path.join(base_dir, file_obj.name)
    with open(file_path, "wb") as f: f.write(file_obj.getbuffer())
    
    text = load_document(file_path)
    if not text: raise ValueError("Empty text.")
    
    asyncio.run(pipeline_orchestrator(text, base_dir, file_obj.name))
    
    return clean_title