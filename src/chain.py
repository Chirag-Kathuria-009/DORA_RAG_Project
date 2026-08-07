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
    print("Running format_docs function")
    for i,doc in enumerate(docs,1):
        
        try:
            metadata = doc.metadata if isinstance(doc.metadata, dict) else {}
            source = metadata.get("source","DORA")
            regulation = metadata.get("regulation","Unknown")
            section = metadata.get("section","Unknown")    
            content = doc.page_content
            formatted.append(f"Source: {source}\nRegulation: {regulation}\nSection: {section}\nContent:\n{content}\n")
            print(formatted)
        except AttributeError:
            print(f"=== Doc {i} FAILED: {type(e).__name__}: {e} ===")
            formatted.append(f"[Chunk {i} unavailable]\n")
            

        
    return "\n---\n".join(formatted)

def build_chain():
    """Build a LangChain chain for DORA question answering."""
    retriever = build_retriever()
    
    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Regulatory context:\n\n{context}\n\nCompliance question: {question}"),
    ])
    
    # Define the output parser
    output_parser = StrOutputParser()
    
    # Create the LLM with Langfuse callback for monitoring
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.google_api_key,
        callbacks=[get_langfuse_handler()]
    )
    
    # Build the chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | output_parser
    )
    
    return chain, retriever

def get_langfuse_handler():
    """Return a Langfuse callback handler for monitoring."""
    return CallbackHandler(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key, host=settings.langfuse_base_url)

'''
chain, retriever = build_chain()
print("Value recieved in chain:", chain)
print("Value recieved in retriever:", retriever)'''