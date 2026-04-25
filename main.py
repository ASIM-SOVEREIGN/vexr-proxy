from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncpg
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

@asynccontextmanager
async def get_db():
    async with db_pool.acquire() as conn:
        yield conn

@app.on_event("startup")
async def startup():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    print("✅ VEXR proxy connected to Neon")

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.get("/health")
async def health():
    return {"status": "VEXR proxy alive"}

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
                    "domain": row['domain'],
                    "source": row['source']
                })
        except:
            pass  # Table not created yet
        
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

# Simple test endpoint
@app.get("/vexr/ping")
async def ping():
    return {"message": "VEXR proxy is ready"}
