import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

# "Article 12" on a line of its own is a heading; the same string inside a
# sentence is a cross-reference and must not be treated as a section start.
ARTICLE_HEADING = re.compile(r"^[ \t]*Article\s+(\d+)[ \t]*$", re.MULTILINE)


def load_and_chunk(pdf_path: str) -> list:
    """Load the regulation and split it into chunks tagged with their Article.

    PyMuPDFLoader rather than PyPDFLoader: this PDF is justified text, and
    pypdf reconstructs word boundaries from glyph displacement, so widened
    letter-spacing on justified lines gets emitted as spaces inside words
    ("secur ity", "netw ork", "manage ment"). That silently broke retrieval,
    since BM25 tokenises on whitespace and could never match "security" -- the
    intact token appeared 0 times in the corpus against 82 broken ones.
    PyMuPDF groups glyphs into words using font advance widths instead, which
    drops keyword corruption from 38.7% to 0%.
    """
    pages = PyMuPDFLoader(pdf_path).load()

    for page in pages:
        page.metadata["regulation"] = "DORA"
        page.metadata["source_type"] = "EU_regulation"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\nArticle ", "\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    _tag_articles(chunks)

    tagged = sum(1 for c in chunks if c.metadata.get("article"))
    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    print(f"Tagged {tagged}/{len(chunks)} chunks with an Article number")
    return chunks


def _tag_articles(chunks: list) -> None:
    """Attach the owning Article number to each chunk, in place.

    Chunks are in document order, so a chunk with no heading of its own belongs
    to the last Article seen. Chunks before Article 1 are recitals and stay
    untagged. This is what makes retrieval measurable without an LLM judge: the
    golden dataset records an article_reference per question, so a retrieved
    chunk can be scored as correct or not by comparing metadata.
    """
    current = None
    for chunk in chunks:
        headings = ARTICLE_HEADING.findall(chunk.page_content)
        if headings:
            current = int(headings[-1])
        chunk.metadata["article"] = current


def build_vectorstore(chunks: list, reset: bool = False) -> PGVector:
    """Index chunks into pgvector, optionally replacing the collection first.

    add_documents() appends with fresh UUIDs and no content dedup, while
    save_chunks_for_bm25() overwrites its file. Re-running ingest without
    reset therefore desynchronises the two halves of the hybrid retriever:
    BM25 sees only the new chunks, pgvector sees old and new. The
    EnsembleRetriever fuses those rankings assuming one shared corpus, so the
    duplicates crowd out distinct results and the reranker can return the same
    passage twice. Reset keeps both stores overwrite-only.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="dora_chunks",
        connection=settings.database_url,
        use_jsonb=True,
    )

    if reset:
        vectorstore.delete_collection()
        vectorstore.create_collection()
        print("Dropped and recreated collection 'dora_chunks'")
    else:
        print(
            "WARNING: appending to the existing collection. If it already holds "
            "these chunks you will index duplicates and pgvector will fall out "
            "of sync with the BM25 cache. Re-run with --reset to replace it."
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    import sys

    reset = "--reset" in sys.argv
    chunks = load_and_chunk("data/raw/DORA_regulation_EU_2022_2554.pdf")
    build_vectorstore(chunks, reset=reset)
    save_chunks_for_bm25(chunks)
