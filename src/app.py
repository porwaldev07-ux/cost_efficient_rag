import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from src.config import settings

app = FastAPI(title="Cost-Efficient RAG Application Engine")


qdrant_client = QdrantClient(path=settings.QDRANT_PATH)


print("Loading local free embedding model for querying...")
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

COLLECTION_NAME = "document_corpus"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    file_type_filter: str = "md"  

class QueryResponse(BaseModel):
    answer: str
    chunks_used: int
    latency_ms: float
    citations: list[str]

@app.post("/query", response_model=QueryResponse)
def handle_rag_query(payload: QueryRequest):
    start_time = time.perf_counter()
    
    
    query_vector = embedding_model.encode(payload.question).tolist()
    
    
    search_response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="file_type", match=MatchValue(value=payload.file_type_filter))]
        ),
        limit=payload.top_k
    )
    search_results = search_response.points
    
    
    SCORE_THRESHOLD = 0.30  
    valid_results = [res for res in search_results if res.score >= SCORE_THRESHOLD]
    
    execution_time = (time.perf_counter() - start_time) * 1000
    
    if not valid_results:
        return QueryResponse(
            answer="I do not have sufficient relevant context within my system to accurately verify an answer.",
            chunks_used=0,
            latency_ms=execution_time,
            citations=[]
        )
    
    
    best_match = valid_results[0]
    answer_text = f"Based on our internal documents: {best_match.payload['text']}"
    
    citations = [res.payload['source'] for res in valid_results]
    

    print(f"[LOG] Latency: {execution_time:.2f}ms | Chunks Used: {len(valid_results)} | Source: {citations}")
    
    return QueryResponse(
        answer=answer_text,
        chunks_used=len(valid_results),
        latency_ms=execution_time,
        citations=list(set(citations))
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)