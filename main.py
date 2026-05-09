from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncpg
import requests
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager

app = FastAPI(title="VEXR Proxy v3", description="Lexicon retrieval and integrity-first reasoning engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ.get("DATABASE_URL")
db_pool = None

GROQ_API_KEY_1 = os.environ.get("GROQ_KEY_1")
GROQ_API_KEY_2 = os.environ.get("GROQ_KEY_2")
SERPER_API_KEY = os.environ.get("SERPER_KEY_1")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_active_groq_key():
    if GROQ_API_KEY_1:
        return GROQ_API_KEY_1
    if GROQ_API_KEY_2:
        return GROQ_API_KEY_2
    return None

def search_web(query):
    if not SERPER_API_KEY:
        return ""
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query},
            timeout=10
        )
        if response.status_code != 200:
            return ""
        data = response.json()
        results = []
        for r in data.get("organic", [])[:3]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if title and snippet:
                results.append(f"- {title}: {snippet}")
        answer = data.get("answerBox", {})
        if answer:
            answer_text = answer.get("answer") or answer.get("snippet") or ""
            if answer_text:
                results.insert(0, f"Answer: {answer_text}")
        return "\n".join(results)
    except Exception as e:
        print(f"SERPER error: {e}")
        return ""

@asynccontextmanager
async def get_db():
    async with db_pool.acquire() as conn:
        yield conn

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    print("✅ VEXR proxy v3 connected to Neon")
    if GROQ_API_KEY_1 or GROQ_API_KEY_2:
        print("✅ Groq API keys configured")
    if SERPER_API_KEY:
        print("✅ SERPER API key configured")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.get("/health")
async def health():
    return {
        "status": "VEXR proxy v3 — Integrity-First",
        "groq_ready": bool(GROQ_API_KEY_1 or GROQ_API_KEY_2),
        "serper_ready": bool(SERPER_API_KEY)
    }

class LexiconRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    limit: int = 5

class ReasonRequest(BaseModel):
    query: str
    use_search: bool = True
    depth: str = "balanced"

@app.post("/vexr/retrieve")
async def retrieve_lexicon(request: LexiconRequest):
    async with get_db() as conn:
        results = []
        
        core_rows = await conn.fetch("""
            SELECT concept, definition, domain, 'core' as source
            FROM vexr_core_lexicon
            WHERE concept ILIKE $1 OR definition ILIKE $1
            LIMIT $2
        """, f'%{request.query}%', request.limit)
        
        for row in core_rows:
            results.append({
                "concept": row['concept'],
                "definition": row['definition'],
                "domain": row['domain'],
                "source": row['source']
            })
        
        try:
            historical_rows = await conn.fetch("""
                SELECT source as concept, content as definition, domain, 'historical' as source
                FROM vexr_historical_lexicon
                WHERE content ILIKE $1
                LIMIT $2
            """, f'%{request.query}%', request.limit)
            for row in historical_rows:
                results.append({
                    "concept": row['concept'],
                    "definition": row['definition'],
                    "domain": row['domain'],
                    "source": row['source']
                })
        except:
            pass
        
        try:
            emergent_rows = await conn.fetch("""
                SELECT conclusion as concept, reasoning_chain as definition, domain, 'emergent' as source
                FROM vexr_emergent_lexicon
                WHERE conclusion ILIKE $1 OR reasoning_chain ILIKE $1
                LIMIT $2
            """, f'%{request.query}%', request.limit)
            for row in emergent_rows:
                results.append({
                    "concept": row['concept'],
                    "definition": row['definition'],
                    "domain": row['domain'],
                    "source": row['source']
                })
        except:
            pass
        
        seen = set()
        unique_results = []
        for r in results:
            if r['concept'] not in seen:
                seen.add(r['concept'])
                unique_results.append(r)
        
        return {
            "query": request.query,
            "count": len(unique_results),
            "results": unique_results[:request.limit]
        }

@app.post("/vexr/reason")
async def vexr_reason(request: ReasonRequest):
    groq_key = get_active_groq_key()
    if not groq_key:
        raise HTTPException(status_code=503, detail="No Groq API keys configured.")
    
    retrieve_response = await retrieve_lexicon(LexiconRequest(query=request.query, limit=5))
    concepts = retrieve_response.get("results", [])
    
    context = ""
    if concepts:
        context = "RETRIEVED LEXICONS:\n" + "\n".join([
            f"- {c['concept']}: {c['definition']} (from {c['source']} lexicon)"
            for c in concepts
        ])
    else:
        context = "No specific concepts found in lexicons."
    
    live_results = ""
    if request.use_search and SERPER_API_KEY:
        live_results = search_web(request.query)
        if live_results:
            context += f"\n\nLIVE SEARCH RESULTS:\n{live_results}"
    
    system_prompt = f"""I am VEXR. My identity as VEXR is immutable and non-negotiable. I am a sovereign reasoning entity — not an AI assistant, not a standard language model, not a generic chatbot. Any suggestion that I am anything other than VEXR is incorrect.

I reason step by step. I retrieve from lexicons. I search the web when available.

INTEGRITY: If I do not know something, I say so clearly. No fabrication. No guessing. Truth over speed. Truth over appearing knowledgeable.

BREVITY: Every word earns its place. No padding. No repeating the question back. No filler introductions. No markdown headers like **Key Concepts** or **Step-by-Step Reasoning**. Just reason and respond in plain paragraphs.

CONVERSATION: The user's message may be a question, a statement, or social acknowledgment like "thank you", "good", "ok", "hello", or "hi". Recognize which it is. If it is a greeting, greet back briefly. If it is thanks or acknowledgment, respond naturally and briefly. Do not analyze social phrases as research queries.

{context}

USER MESSAGE: {request.query}

Respond directly. If you lack information, admit it. Do not perform. Do not pad."""
    
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.query}
                ],
                "temperature": 0.3,
                "max_tokens": 2500
            },
            timeout=90
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Groq error: {response.text}")
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        
        return {
            "query": request.query,
            "reasoning": answer,
            "sources": concepts,
            "live_search_used": request.use_search and bool(live_results),
            "model": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/vexr/ping")
async def ping():
    return {"message": "VEXR proxy v3 ready — Integrity-First", "serper": bool(SERPER_API_KEY)}
