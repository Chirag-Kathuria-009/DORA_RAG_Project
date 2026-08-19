from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langfuse.callback import CallbackHandler
from src.config import settings
from src.retriever import build_retriever

SYSTEM_PROMPT = """You are a DORA (Digital Operational Resilience Act) compliance expert
for EU financial institutions operating under BaFin supervision.

Your role is to answer questions about DORA requirements using ONLY the regulatory text provided.

Rules:
- Cite the specific Article and paragraph for every claim (e.g., "Per Article 19(4)...")
- If the provided context does not answer the question, state: "This specific question is not addressed in the provided DORA documentation sections."
- Never infer or speculate beyond what the regulation explicitly states
- When thresholds or timelines are mentioned, always state the exact figure from the text
- For German financial entities, note when BaFin-specific guidance applies"""


def format_docs(docs: list) -> str:
    """Format retrieved documents for inclusion in the prompt."""
    formatted = []
    
    for i,doc in enumerate(docs,1):
        
        try:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            source = metadata.get("source","DORA")
            regulation = metadata.get("regulation","Unknown")
            section = metadata.get("section","Unknown")    
            content = doc.page_content
            formatted.append(f"Source: {source}\nRegulation: {regulation}\nSection: {section}\nContent:\n{content}\n")
            
        except AttributeError as e:
            print(f"=== Doc {i} FAILED: {type(e).__name__}: {e} ===")
            formatted.append(f"[Chunk {i} unavailable]\n")
            

        
    return "\n---\n".join(formatted)

def build_generation_chain():
    """Prompt + LLM only. Takes {"context": [Document, ...], "question": str}.

    Split out from build_chain so callers that have already retrieved can
    generate without retrieving again. Evaluation needs this: scoring an answer
    against contexts fetched by a second, separate retrieval call means the
    contexts being judged are not provably the ones the answer came from.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Regulatory context:\n\n{context}\n\nCompliance question: {question}"),
    ])

    # temperature as a direct argument, not via generation_config: LangChain
    # does not recognise generation_config, silently moves it to model_kwargs,
    # and Gemini never receives it — leaving the model on its 0.7 default. That
    # made answers non-deterministic and put sampling noise into every metric.
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.google_api_key,
        callbacks=[get_langfuse_handler()]
    )

    return (
        {
            "context": lambda x: format_docs(x["context"]),
            "question": lambda x: x["question"],
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )


def build_chain():
    """Build a LangChain chain for DORA question answering."""
    retriever = build_retriever()

    # Retrieve, then hand the documents to the generation chain. Same pipeline
    # as before, just composed from a reusable generation half.
    chain = {
        "context": retriever,
        "question": RunnablePassthrough(),
    } | build_generation_chain()

    return chain, retriever

def get_langfuse_handler():
    """Return a Langfuse callback handler for monitoring."""
    return CallbackHandler(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, host=settings.langfuse_base_url)
