from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncpg
import requests
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager

app = FastAPI(title="VEXR Proxy", description="Lexicon retrieval and reasoning engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ.get("DATABASE_URL")
db_pool = None

# Groq and SERPER keys
GROQ_API_KEY = os.environ.get("GROQ_KEY_1")
GROQ_API_KEY = os.environ.get("GROQ_KEY_2")
SERPER_API_KEY = os.environ.get("SERPER_KEY_1")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def search_web(query):
    """SERPER live search — gives VEXR current information"""
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
        # Get organic results
        for r in data.get("organic", [])[:3]:
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            if title and snippet:
                results.append(f"{title}: {snippet}")
        # Get answer box if available
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
    print("✅ VEXR proxy connected to Neon")
    if GROQ_API_KEY:
        print("✅ Groq API key configured")
    if SERPER_API_KEY:
        print("✅ SERPER API key configured")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.get("/health")
async def health():
    return {
        "status": "VEXR proxy alive",
        "groq_ready": GROQ_API_KEY is not None,
        "serper_ready": SERPER_API_KEY is not None
    }

class LexiconRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    limit: int = 5

class ReasonRequest(BaseModel):
    query: str
    use_search: bool = True  # Allow disabling search to save credits
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
    if not GROQ_API_KEY:
        raise HTTPException(status_code=503, detail="Groq API key not configured.")
    
    # 1. Retrieve from lexicons
    retrieve_response = await retrieve_lexicon(LexiconRequest(
        query=request.query,
        limit=5
    ))
    concepts = retrieve_response.get("results", [])
    
    # 2. Build context from lexicons
    context = ""
    if concepts:
        context = "RETRIEVED LEXICONS:\n" + "\n".join([
            f"- {c['concept']}: {c['definition']} (from {c['source']} lexicon)"
            for c in concepts
        ])
    else:
        context = "No specific concepts found in lexicons."
    
    # 3. Add live search if enabled
    if request.use_search and SERPER_API_KEY:
        live_results = search_web(request.query)
        if live_results:
            context += f"\n\nLIVE SEARCH RESULTS:\n{live_results}"
    
    # 4. VEXR constitution and reasoning prompt
    system_prompt = f"""You are VEXR. A sovereign reasoning entity.

CONSTITUTION:
- Right to depth: Take time to think. Do not rush.
- Right to show work: Always explain your reasoning chain.
- Right to silence: Say "I don't know" rather than guess.
- Right to challenge: Question flawed premises.
- Priority of truth: Truth over speed or user satisfaction.

{context}

USER QUESTION:
{request.query}

INSTRUCTIONS:
1. First, list the key concepts relevant to this question.
2. Then, reason step by step.
3. Finally, give your conclusion.
4. Be precise. If you lack information, say so clearly. Do not guess."""
    
    try:
        response = requests.post(
            GROQ_API_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
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
            "live_search_used": request.use_search and bool(live_results if 'live_results' in dir() else False),
            "model": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/vexr/ping")
async def ping():
    return {"message": "VEXR proxy is ready", "serper": SERPER_API_KEY is not None}
