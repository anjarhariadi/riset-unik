import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import AnalyzeResponse, AnalyzeRequest
from services.doaj import search_doaj
from services.openalex import search_openalex
from services.semantic_scholar import search_s2
from services.similiarity import analyze_similarity
from services.topic_generator import generate_topic
from slowapi import Limiter
from slowapi.util import get_remote_address

from utils.similarity import get_avg_similarity_score
from utils.validator import validate_topic, validate_topic_yapping


limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

# 🛡️ Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "https://risetunik.algieba.top", "http://risetunik.algieba.top"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@limiter.limit("2/minute")
async def root(request: Request):
    return {"message": "Hello World"}

@app.post("/analyze-topic", response_model=AnalyzeResponse)
@limiter.limit("20/minute")
async def analyze_topic(data: AnalyzeRequest, request: Request):
    topic = data.topic
    is_valid, result = validate_topic(topic)
    if not is_valid:
        raise HTTPException(status_code=400, detail=result)

    if not topic:
        raise HTTPException(status_code=400, detail="Topik tidak boleh kosong.")

    # Use asyncio.gather to send requests concurrently
    doaj_results, openalex_results, s2_results = await asyncio.gather(
        search_doaj(topic), search_openalex(topic, 40), search_s2(topic, 40)
    )

    results = doaj_results + openalex_results + s2_results

    # Flatten all paper title
    titles = [{"title": item["title"], "link": item["link"]} for item in results if item]

    result = analyze_similarity(data.topic, titles)
    average_similarity = get_avg_similarity_score(result)

    return AnalyzeResponse(average_similarity=average_similarity, results=result)

@app.post("/analyze-yapping", response_model=AnalyzeResponse)
@limiter.limit("20/minute")
async def analyze_topic(data: AnalyzeRequest, request: Request):
    topic_desc = data.topic.strip()
    is_valid, result = validate_topic_yapping(topic_desc)
    if not is_valid:
        raise HTTPException(status_code=400, detail=result)

    topics = generate_topic(topic_desc)
    print(topics)
    if topics is None or topics.topic_id is None or topics.topic_en is None:
        raise HTTPException(status_code=500, detail="Terjadi kesalahan pada server.")

    async def fetch_results(topic):
        doaj_results, openalex_results, s2_results = await asyncio.gather(
            search_doaj(topic), search_openalex(topic), search_s2(topic)
        )
        return doaj_results + openalex_results + s2_results

    results_id, results_en = await asyncio.gather(
        fetch_results(topics.topic_id),
        fetch_results(topics.topic_en)
    )

    def extract_titles(results: list):
        return [
            {"title": item.get("title", "No Title"), "link": item.get("link", "#")}
            for item in results
        ]

    titles_id = extract_titles(results_id)
    titles_en = extract_titles(results_en)

    result_id, result_en = await asyncio.gather(
        asyncio.to_thread(analyze_similarity, topics.topic_id, titles_id),
        asyncio.to_thread(analyze_similarity, topics.topic_en, titles_en)
    )

    titles = result_id + result_en
    average_similarity = get_avg_similarity_score(titles)

    return AnalyzeResponse(average_similarity=average_similarity, results=titles)