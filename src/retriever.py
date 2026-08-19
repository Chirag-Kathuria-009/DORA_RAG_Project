import json
from functools import lru_cache
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain.schema import Document
from src.config import settings
from langchain.retrievers.document_compressors import CrossEncoderReranker


def load_chunks_from_cache(path: str = "data/chunks_cache.json") -> list[Document]:
    """BM25Retriever loads from documents in memory — load chunks from cache."""
    with open(path, "r") as f:
        serialized = json.load(f)
    chunks = [
        Document(page_content=c["page_content"], metadata=c["metadata"])
        for c in serialized
    ]
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks

@lru_cache(maxsize=2)
def _cross_encoder(model_name: str) -> HuggingFaceCrossEncoder:
    """Cross-encoders are large (bge-reranker-large is ~1.6GB) and stateless
    once loaded. Cached so an evaluation sweep over configs doesn't reload the
    model for every variant it tests."""
    return HuggingFaceCrossEncoder(model_name=model_name)


def build_stages(
    top_k: int | None = None,
    rerank_top_n: int | None = None,
    bm25_weight: float = 0.4,
    reranker_model: str = "BAAI/bge-reranker-large",
) -> dict:
    """Build the pipeline and return each stage separately.

    build_retriever() returns only the final reranked output, which is what the
    API needs but not enough to debug a miss: it can't distinguish "BM25 and
    pgvector never found the chunk" from "they found it and the cross-encoder
    ranked it out". Those have opposite fixes, so the evaluation harness scores
    each stage against the same questions.
    """
    top_k = top_k or settings.retrieval_top_k
    rerank_top_n = rerank_top_n or settings.rerank_top_n

    chunks = load_chunks_from_cache()

    bm25_retriever = BM25Retriever.from_documents(documents=chunks, k=top_k)

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="dora_chunks",
        connection=settings.database_url,
        use_jsonb=True,
    )
    pgvector_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pgvector_retriever],
        weights=[bm25_weight, 1.0 - bm25_weight],
    )
    ### removing usage of CohereRerank for now, as it have limit of api call
    '''reranker = CohereRerank(
        model="rerank-english-v3.0",
        top_n=settings.rerank_top_n,
        cohere_api_key=settings.cohere_api_key
    )'''

    stages = {
        "bm25": bm25_retriever,
        "vector": pgvector_retriever,
        "hybrid": hybrid_retriever,
    }

    if reranker_model:
        stages["reranked"] = ContextualCompressionRetriever(
            base_compressor=CrossEncoderReranker(
                model=_cross_encoder(reranker_model), top_n=rerank_top_n
            ),
            base_retriever=hybrid_retriever,
        )

    return stages


def build_retriever(**kwargs) -> ContextualCompressionRetriever:
    """Full pipeline: hybrid BM25 + pgvector retrieval, then cross-encoder rerank."""
    return build_stages(**kwargs)["reranked"]

#input_type=HuggingFaceCrossEncoder