import json
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
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

def build_retriever() -> ContextualCompressionRetriever:
    """Build a retriever that combines BM25 and pgvector retrieval, with reranking."""
    # Load chunks from cache for BM25
    chunks = load_chunks_from_cache()

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(documents=chunks, k=settings.retrieval_top_k)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="dora_chunks",
        connection=settings.database_url,
        use_jsonb=True,
    )
    
    # Create pgvector retriever
    pgvector_retriever = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_top_k})
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pgvector_retriever],
        weights=[0.4, 0.6]
    )
    ### removing usage of CohereRerank for now, as it have limit of api call    
    '''reranker = CohereRerank(
        model="rerank-english-v3.0",  
        top_n=settings.rerank_top_n,
        cohere_api_key=settings.cohere_api_key
    )'''
    
    rerank_model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        device="cpu"
    )
    
    reranker = CrossEncoderReranker(
        cross_encoder=rerank_model,
        top_n=settings.rerank_top_n
    )
    
    # Wrap the hybrid retriever with a ContextualCompressionRetriever for reranking
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever
    )
    return compressed_retriever
