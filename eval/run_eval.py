import json
import time
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from datasets import Dataset
from src.chain import build_chain
from src.retriever import build_retriever
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import settings


## Initial declarations

rate_limit_delay = 20
checkpoint_path = Path("data/eval/eval_checkpoint.json")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)

def safe_invoke(runnable,query):
    """Invoke the chain with retry logic for rate limiting."""
    return runnable.invoke(query)

def collect_predictions(chain, retriever, golden_dataset):
    #run rag pipeline over golden dataset with checkpointing to allow resuming after interruptions
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            records = json.load(f)
    else:
        records = []
    
    completed = {r["question"] for r in records}
    
    for i,entry in enumerate(golden_dataset):
        question = entry.get("question")
        if question in completed:
            print(f"Skipping already processed question: {question}")
            continue
        
        try:
            answer = safe_invoke(chain, question)
            docs = safe_invoke(retriever, question)
            context = [doc.page_content for doc in docs]
            
            records.append({
                "question": question,
                "answer": answer,
                "context": context,
                "ground_truth": entry.get("ground_truth")
            })
            
            
            
            print(f"Processed {i+1}/{len(golden_dataset)}: {question}")
        
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            break
        
        # Save checkpoint after each successful prediction
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(records, f, indent=4)
        
        time.sleep(rate_limit_delay)  # Delay to respect rate limits
    
    return records
def run_eval():
    """Run evaluation on the golden dataset."""
    chain, retriever = build_chain()
    
    with open("data/eval/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    predictions = collect_predictions(chain, retriever, golden_dataset)
    
    if(len(predictions)<len(golden_dataset)):
        print(f"Warning: Only {len(predictions)} out of {len(golden_dataset)} questions were processed. Check the checkpoint file for details.")
    
    
        
    dataset = Dataset.from_dict({
        "question": [p["question"] for p in predictions],
        "answer": [p["answer"] for p in predictions],
        "retrieved_contexts": [p["context"] for p in predictions],
        "ground_truth": [p["ground_truth"] for p in predictions]
    })
    
    judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=settings.llm_model,
        api_key=settings.google_api_key
    ))
    
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    ))
    
    print("Starting evaluation with RAGAS...")
    result = evaluate(
        dataset,
        llm=judge_llm,
        embeddings=judge_embeddings,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        raise_exceptions=False)
    
    
    
    df = result.to_pandas()
    #df["retrieved_contexts"] = df["context"]
    scores = df[[metric.name for metric in [faithfulness, answer_relevancy, context_precision, context_recall]]].mean().to_dict()
    for metric, score in scores.items():
        print(f"  {metric:20} {score:.4f}")
        
    
    print("Evaluation completed. Average scores:")
    
    
    with open("data/eval/evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    df.to_csv("data/eval/per_question_results.csv", index=False)
    print("\nSaved: evaluation_results.json + per_question_results.csv")
    return scores

if __name__ == "__main__":
    run_eval()