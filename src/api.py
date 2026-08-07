from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from src.config import settings
from src.chain import build_chain, get_langfuse_handler
import time
app = FastAPI(title="DORA RAG API",
              description="Regulatory Q&A on DORA (EU 2022/2554) with hybrid retrieval and citation",
              version="1.0.0")

chain, retriever = build_chain()

class QueryRequest(BaseModel):
    question: str
    trace_id: bool = True

class SourceChunk(BaseModel):
    source: str
    page: str
    content: str
    relevance_rank: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    model_used: str
    

@app.post("/query", response_model=QueryResponse)
def query_dora(request: QueryRequest):
    try:
        start = time.time()
        callbacks = [get_langfuse_handler()] if request.trace_id else []

        
        answer = chain.invoke(
            request.question,
            config={"callbacks": callbacks}
        )

        
        model_used = settings.llm_model

        
        source_docs = retriever.invoke(request.question)
        source_chunks = [
            SourceChunk(
                source=str(doc.metadata.get("source", "DORA")),
                page=str(doc.metadata.get("page", "?")),
                content=doc.page_content[:300],
                relevance_rank=i + 1,
            )
            for i, doc in enumerate(source_docs)
        ]

        latency_ms = float((time.time() - start) * 1000)

        return QueryResponse(
            answer=answer,
            sources=source_chunks,
            latency_ms=latency_ms,
            model_used=model_used,
        )

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "service": "DORA RAG API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



