import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from src.config import settings
from langchain_huggingface import HuggingFaceEmbeddings

def load_and_chunk(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Add DORA-specific metadata to every chunk
    for page in pages:
        page.metadata["regulation"] = "DORA"
        page.metadata["source_type"] = "EU_regulation"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\nArticle ", "\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks


def build_vectorstore(chunks: list) -> PGVector:
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="dora_chunks",
        connection=settings.database_url,
        use_jsonb=True,
    )
    vectorstore.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into pgvector")
    return vectorstore


def save_chunks_for_bm25(chunks: list, path: str = "data/chunks_cache.json"):
    """BM25Retriever loads from documents in memory — save chunks so we
    can reload them at app startup without re-processing the PDF."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    with open(path, "w") as f:
        json.dump(serialized, f)
    print(f"Saved {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    chunks = load_and_chunk("data/raw/DORA_regulation_EU_2022_2554.pdf")
    build_vectorstore(chunks)
    save_chunks_for_bm25(chunks)