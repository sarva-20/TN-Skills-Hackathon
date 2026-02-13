"""
StartupTN Hackathon - PRODUCTION-READY Backend
===============================================
Enhanced Government Scheme Mapper with:
- Request caching for speed
- Error handling & logging
- Advanced ML matching
- API statistics

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from googletrans import Translator
from functools import lru_cache
import logging
from datetime import datetime

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="TN Scheme Mapper API - Production",
    description="AI-Powered Government Scheme Mapping for Tamil Nadu",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS - More restrictive for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with actual frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request tracking
request_count = {"total": 0, "match": 0, "search": 0, "schemes": 0}
request_times = []

# Load ML model
logger.info("🤖 Loading ML model...")
try:
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    translator = Translator()
    logger.info("✅ ML model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load ML model: {e}")
    raise

# ============================================================================
# ENHANCED DATA MODELS
# ============================================================================

class Eligibility(BaseModel):
    income: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    caste: Optional[str] = None
    residence: Optional[str] = None
    other: Optional[str] = None

class Scheme(BaseModel):
    id: str
    name: str
    name_tamil: Optional[str] = ""
    category: str
    department: str
    description: str
    description_tamil: Optional[str] = ""
    benefits: str
    eligibility: Dict[str, Any]
    documents_required: str
    application_process: str
    application_link: Optional[str] = ""
    contact: str

class UserInput(BaseModel):
    # Form-based input
    age: Optional[int] = Field(None, ge=0, le=120, description="User's age")
    gender: Optional[str] = Field(None, pattern="^(Male|Female|Other)$")
    annual_income: Optional[int] = Field(None, ge=0, description="Annual income in rupees")
    caste: Optional[str] = None
    occupation: Optional[str] = None
    location: Optional[str] = "Tamil Nadu"
    
    # Conversational input
    query: Optional[str] = Field(None, max_length=500)
    language: str = Field("en", pattern="^(en|ta)$")

class MatchedScheme(BaseModel):
    scheme: Scheme
    match_score: float = Field(..., ge=0, le=100)
    match_reason: str

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    total_schemes: int
    uptime_seconds: float
    total_requests: int

# ============================================================================
# IN-MEMORY DATABASE WITH CACHING
# ============================================================================

SCHEMES_DB: List[Scheme] = []
SCHEME_EMBEDDINGS = None
START_TIME = time.time()

@lru_cache(maxsize=1)
def get_schemes_cache():
    """Cached scheme retrieval"""
    return SCHEMES_DB

def load_schemes():
    """Load schemes from JSON file with error handling"""
    global SCHEMES_DB, SCHEME_EMBEDDINGS
    
    possible_paths = [
        Path("../data/tn_schemes_starter.json"),
        Path("data/tn_schemes_starter.json"),
        Path("tn_schemes_starter.json"),
    ]
    
    for json_path in possible_paths:
        if json_path.exists():
            logger.info(f"📂 Loading schemes from: {json_path}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    schemes_data = json.load(f)
                
                SCHEMES_DB = [Scheme(**scheme) for scheme in schemes_data]
                logger.info(f"✅ Loaded {len(SCHEMES_DB)} schemes successfully!")
                
                # Generate embeddings
                logger.info("🧠 Generating ML embeddings...")
                scheme_texts = [
                    f"{s.name} {s.category} {s.description} {s.benefits}"
                    for s in SCHEMES_DB
                ]
                SCHEME_EMBEDDINGS = model.encode(scheme_texts)
                logger.info("✅ Embeddings ready!")
                return
                
            except Exception as e:
                logger.error(f"❌ Error loading schemes: {e}")
                continue
    
    logger.warning("⚠️ No schemes file found! Starting with empty database.")

# ============================================================================
# ENHANCED MATCHING LOGIC
# ============================================================================

def calculate_eligibility_score(user: UserInput, scheme: Scheme) -> tuple[float, str]:
    """
    Advanced eligibility scoring with detailed reasoning
    Returns: (score 0-100, reason string)
    """
    score = 0.0
    max_score = 0.0
    reasons = []
    
    eligibility = scheme.eligibility
    
    # Age matching (20 points)
    if eligibility.get('age') and user.age:
        max_score += 20
        age_criteria = eligibility['age'].lower()
        
        try:
            if 'above' in age_criteria or 'and above' in age_criteria:
                min_age = int(''.join(filter(str.isdigit, age_criteria.split('above')[0])))
                if user.age >= min_age:
                    score += 20
                    reasons.append(f"Age ✓ ({user.age} ≥ {min_age})")
                else:
                    reasons.append(f"Age ✗ ({user.age} < {min_age})")
            elif 'to' in age_criteria or '-' in age_criteria:
                ages = [int(x) for x in age_criteria.split() if x.isdigit()]
                if len(ages) >= 2 and ages[0] <= user.age <= ages[1]:
                    score += 20
                    reasons.append(f"Age ✓ ({ages[0]}-{ages[1]})")
                else:
                    reasons.append(f"Age ✗ (not in {ages[0]}-{ages[1]})")
            else:
                score += 10  # Partial credit
        except:
            score += 5
    
    # Income matching (30 points) - Most important!
    if eligibility.get('income') and user.annual_income:
        max_score += 30
        income_criteria = eligibility['income'].lower()
        
        try:
            if 'less than' in income_criteria or 'below' in income_criteria:
                income_limit = int(''.join(filter(str.isdigit, income_criteria.replace(',', ''))))
                if user.annual_income <= income_limit:
                    score += 30
                    reasons.append(f"Income ✓ (₹{user.annual_income:,} ≤ ₹{income_limit:,})")
                else:
                    reasons.append(f"Income ✗ (₹{user.annual_income:,} > ₹{income_limit:,})")
            elif 'between' in income_criteria:
                nums = [int(x.replace(',', '')) for x in income_criteria.split() if x.replace(',', '').isdigit()]
                if len(nums) >= 2 and nums[0] <= user.annual_income <= nums[1]:
                    score += 30
                    reasons.append(f"Income ✓ (in range)")
                else:
                    reasons.append(f"Income ✗ (out of range)")
            else:
                score += 15  # Partial credit
        except:
            score += 10
    
    # Gender matching (20 points)
    if eligibility.get('gender') and user.gender:
        max_score += 20
        if eligibility['gender'].lower() == user.gender.lower():
            score += 20
            reasons.append("Gender ✓")
        else:
            reasons.append("Gender ✗")
    elif not eligibility.get('gender'):
        max_score += 20
        score += 10  # Bonus for no restriction
        reasons.append("Gender: Any")
    
    # Caste matching (15 points)
    if eligibility.get('caste') and user.caste:
        max_score += 15
        caste_criteria = eligibility['caste'].lower()
        if user.caste.lower() in caste_criteria or 'all' in caste_criteria:
            score += 15
            reasons.append("Category ✓")
        else:
            reasons.append("Category ✗")
    elif not eligibility.get('caste'):
        max_score += 15
        score += 7
        reasons.append("Category: Any")
    
    # Residence (15 points)
    if eligibility.get('residence'):
        max_score += 15
        if 'tamil nadu' in eligibility['residence'].lower():
            score += 15
            reasons.append("Residence ✓ (TN)")
    
    # Normalize to 0-100
    if max_score > 0:
        final_score = (score / max_score) * 100
    else:
        final_score = 50
    
    reason_text = " | ".join(reasons) if reasons else "General eligibility"
    
    return round(final_score, 2), reason_text


def semantic_search(query: str, top_k: int = 10) -> List[tuple[Scheme, float]]:
    """Enhanced semantic search with better scoring"""
    if SCHEME_EMBEDDINGS is None:
        return []
    
    try:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, SCHEME_EMBEDDINGS)[0]
        
        # Get top-k with threshold
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.15:  # Lower threshold for better recall
                results.append((SCHEMES_DB[idx], float(similarities[idx])))
        
        return results
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        return []


def translate_if_needed(text: str, source_lang: str) -> str:
    """Translate Tamil to English with error handling"""
    if source_lang == "ta":
        try:
            translated = translator.translate(text, src='ta', dest='en')
            return translated.text
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text
    return text

# ============================================================================
# MIDDLEWARE FOR REQUEST TRACKING
# ============================================================================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Track metrics
    process_time = time.time() - start_time
    request_times.append(process_time)
    request_count["total"] += 1
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("🚀 Starting TN Scheme Mapper API...")
    load_schemes()
    logger.info("✅ API Ready!")

@app.get("/", response_model=dict)
async def root():
    """API root - health check"""
    return {
        "status": "online",
        "message": "TN Scheme Mapper API - Production Ready",
        "version": "2.0.0",
        "total_schemes": len(SCHEMES_DB),
        "docs": "/docs",
        "endpoints": {
            "match": "/match",
            "schemes": "/schemes",
            "search": "/search",
            "stats": "/stats",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Detailed health check"""
    uptime = time.time() - START_TIME
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        total_schemes=len(SCHEMES_DB),
        uptime_seconds=round(uptime, 2),
        total_requests=request_count["total"]
    )

@app.get("/schemes", response_model=List[Scheme])
async def get_all_schemes(
    category: Optional[str] = None,
    limit: Optional[int] = None
):
    """Get all schemes with optional filtering"""
    request_count["schemes"] += 1
    
    schemes = SCHEMES_DB
    
    if category:
        schemes = [s for s in schemes if s.category.lower() == category.lower()]
    
    if limit:
        schemes = schemes[:limit]
    
    return schemes

@app.get("/schemes/{scheme_id}", response_model=Scheme)
async def get_scheme(scheme_id: str):
    """Get specific scheme by ID"""
    scheme = next((s for s in SCHEMES_DB if s.id == scheme_id), None)
    if not scheme:
        raise HTTPException(status_code=404, detail=f"Scheme {scheme_id} not found")
    return scheme

@app.get("/categories")
async def get_categories():
    """Get all unique categories with counts"""
    category_counts = {}
    for scheme in SCHEMES_DB:
        category_counts[scheme.category] = category_counts.get(scheme.category, 0) + 1
    
    return {
        "categories": sorted(category_counts.keys()),
        "counts": category_counts,
        "total": len(category_counts)
    }

@app.post("/match", response_model=List[MatchedScheme])
async def match_schemes(user_input: UserInput):
    """
    ENHANCED MATCHING ENDPOINT
    
    Two modes:
    1. Form-based: age, income, gender, etc.
    2. Conversational: natural language query
    3. Hybrid: Both combined for best results!
    """
    request_count["match"] += 1
    start_time = time.time()
    
    matched_schemes = []
    
    try:
        # MODE 1: Conversational/NLP matching
        if user_input.query:
            query_english = translate_if_needed(user_input.query, user_input.language)
            search_results = semantic_search(query_english, top_k=15)
            
            for scheme, similarity_score in search_results:
                # Hybrid: Combine semantic + eligibility if form data provided
                if user_input.age or user_input.annual_income:
                    eligibility_score, reason = calculate_eligibility_score(user_input, scheme)
                    final_score = (similarity_score * 100 * 0.5) + (eligibility_score * 0.5)
                    reason = f"Semantic match ({similarity_score:.2f}) + {reason}"
                else:
                    final_score = similarity_score * 100
                    reason = f"Semantic relevance: {similarity_score:.2f}"
                
                if final_score >= 20:  # Lower threshold
                    matched_schemes.append(MatchedScheme(
                        scheme=scheme,
                        match_score=round(final_score, 2),
                        match_reason=reason
                    ))
        
        # MODE 2: Pure form-based matching
        else:
            for scheme in SCHEMES_DB:
                eligibility_score, reason = calculate_eligibility_score(user_input, scheme)
                
                if eligibility_score >= 30:
                    matched_schemes.append(MatchedScheme(
                        scheme=scheme,
                        match_score=eligibility_score,
                        match_reason=reason
                    ))
        
        # Sort and return top 10
        matched_schemes.sort(key=lambda x: x.match_score, reverse=True)
        result = matched_schemes[:10]
        
        process_time = time.time() - start_time
        logger.info(f"Match query processed in {process_time:.3f}s - {len(result)} results")
        
        return result
        
    except Exception as e:
        logger.error(f"Match error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[Scheme])
async def search_schemes(
    query: str,
    language: str = "en",
    limit: int = 10
):
    """Simple search endpoint"""
    request_count["search"] += 1
    
    try:
        query_english = translate_if_needed(query, language)
        results = semantic_search(query_english, top_k=limit)
        return [scheme for scheme, _ in results]
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """API statistics dashboard"""
    avg_response_time = sum(request_times) / len(request_times) if request_times else 0
    
    category_distribution = {}
    for scheme in SCHEMES_DB:
        category_distribution[scheme.category] = category_distribution.get(scheme.category, 0) + 1
    
    return {
        "system": {
            "total_schemes": len(SCHEMES_DB),
            "has_tamil_support": sum(1 for s in SCHEMES_DB if s.name_tamil) > 0,
            "uptime_seconds": round(time.time() - START_TIME, 2),
            "status": "healthy"
        },
        "requests": {
            "total": request_count["total"],
            "match": request_count["match"],
            "search": request_count["search"],
            "schemes": request_count["schemes"],
            "avg_response_time_ms": round(avg_response_time * 1000, 2)
        },
        "schemes": {
            "by_category": category_distribution,
            "categories_count": len(category_distribution)
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc)
        }
    )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=False,  # Disable in production
        log_level="info"
    )