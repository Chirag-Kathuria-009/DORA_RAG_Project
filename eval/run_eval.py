import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainembeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from datasets import Dataset
from src.chain import build_chain
from src.retriever import build_retriever
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential


## Initial declarations

rate_limit_delay = 5
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
            
            # Save checkpoint after each successful prediction
            with open(checkpoint_path, "w") as f:
                json.dump(records, f, indent=4)
            
            print(f"Processed {i+1}/{len(golden_dataset)}: {question}")
        
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
def run_eval():
    """Run evaluation on the golden dataset."""
    chain, retriever = build_chain()
    
    with open("data/eval/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)

    questions,answers,contexts,ground_truths = [],[],[],[]
    print("Preparing evaluation dataset...")
    for entry in golden_dataset:
        q = entry.get("question")
        gt = entry.get("ground_truth")
        
        ans = chain.invoke(q)
        docs = retriever.invoke(q)
        ctxt = [doc.page_content for doc in docs]
        questions.append(q)
        answers.append(ans)
        contexts.append(ctxt)
        ground_truths.append(gt)
        
        print(f"Question: {q}\nAnswer: {ans}\nContext: {ctxt}\nGround Truth: {gt}\n---\n")
        
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "context": contexts,
        "ground_truth": ground_truths
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    
    scores = result.to_pandas().mean().to_dict()
    print("Evaluation completed. Average scores:")
    
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")
    
    with open("data/eval/evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=4)
    
    return scores

if __name__ == "__main__":
    run_eval()