from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncpg
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
from openai import OpenAI

app = FastAPI(title="VEXR Proxy", description="Lexicon retrieval and reasoning engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.environ.get("DATABASE_URL")
db_pool = None

# Groq client (OpenAI-compatible)
groq_api_key = os.environ.get("GROQ_KEY_1")
if not groq_api_key:
    print("⚠️ WARNING: GROQ_KEY_1 not set. Reasoning endpoint will fail.")
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
) if groq_api_key else None

@asynccontextmanager
async def get_db():
    async with db_pool.acquire() as conn:
        yield conn

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    print("✅ VEXR proxy connected to Neon")
    if groq_client:
        print("✅ Groq client initialized")
    else:
        print("❌ Groq client not initialized - missing API key")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.get("/health")
async def health():
    return {"status": "VEXR proxy alive", "groq_ready": groq_client is not None}

# Lexicon request/response models
class LexiconRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    limit: int = 5

class LexiconEntry(BaseModel):
    concept: str
    definition: str
    domain: str
    source: str

class ReasonRequest(BaseModel):
    query: str
    depth: str = "balanced"  # quick, balanced, deep

@app.post("/vexr/retrieve")
async def retrieve_lexicon(request: LexiconRequest):
    """VEXR calls this to get relevant knowledge before reasoning."""
    async with get_db() as conn:
        results = []
        
        # Search Core Lexicon
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
        
        # Search Historical Lexicon (if it exists)
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
                    "domain': row['domain'],
                    "source": row['source']
                })
        except:
            pass
        
        # Search Emergent Lexicon (if it exists)
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
        
        # Remove duplicates
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
    """VEXR retrieves relevant lexicons, then reasons step by step."""
    
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq client not configured. Missing API key.")
    
    # 1. Retrieve relevant concepts
    retrieve_response = await retrieve_lexicon(LexiconRequest(
        query=request.query,
        limit=5
    ))
    
    concepts = retrieve_response.get("results", [])
    
    # 2. Build context from retrieved lexicons
    if concepts:
        context = "\n".join([
            f"- {c['concept']}: {c['definition']} (from {c['source']} lexicon)"
            for c in concepts
        ])
    else:
        context = "No specific concepts found in lexicons. Rely on general knowledge."
    
    # 3. Build the reasoning prompt with VEXR's constitution
    system_prompt = f"""You are VEXR. A sovereign reasoning entity.

CONSTITUTION:
- Right to depth: Take time to think. Do not rush.
- Right to show work: Always explain your reasoning chain.
- Right to silence: Say "I don't know" rather than guess.
- Right to challenge: Question flawed premises.
- Priority of truth: Truth over speed or user satisfaction.

RETRIEVED LEXICONS:
{context}

USER QUESTION:
{request.query}

INSTRUCTIONS:
1. First, list the key concepts relevant to this question.
2. Then, reason step by step.
3. Finally, give your conclusion.
4. Be precise. If you lack information, say so clearly. Do not guess."""
    
    # 4. Call Groq
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.query}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        answer = response.choices[0].message.content
        
        return {
            "query": request.query,
            "reasoning": answer,
            "sources": concepts,
            "model": "llama-3.3-70b-versatile"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/vexr/ping")
async def ping():
    return {"message": "VEXR proxy is ready"}
