import json
import time
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from datasets import Dataset
from src.chain import build_chain
from src.retriever import build_retriever
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import settings
from langchain_groq import ChatGroq
import pandas as pd



## Initial declarations

rate_limit_delay = 20
checkpoint_path = Path("data/eval/eval_checkpoint.json")
judge_model = "llama-3.3-70b-versatile"
metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
OUT_DIR = Path("data/eval")
metric_cache = Path("data/eval/metric_cache.json")

JUDGE_BY_METRIC = {
    "faithfulness":      "llama-3.3-70b-versatile",   
    "context_precision": "openai/gpt-oss-120b",       
    "context_recall":    "openai/gpt-oss-20b",          
    "answer_relevancy":  "qwen/qwen3.6-27b",        
}

def load_metric_cache():
    if metric_cache.exists():
        with open(metric_cache, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_metric_cache(cache):
    metric_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(metric_cache, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

#metrics_name = ["faithfullness", "answer_relevancy", "context_precision", "context_recall"]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)

def safe_invoke(runnable,query):
    """Invoke the chain with retry logic for rate limiting."""
    return runnable.invoke(query)

def sample_records(records,n=None):
    if n is None or n >= len(records):
        return records
    step = len(records) // n
    return [records[i] for i in range(0, len(records), step)][:n]


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
            
            if not context:
                print(f"[{i}] WARNING: no chunks retrieved — skipping")
                continue
            
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
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2,ensure_ascii=False)
        
        time.sleep(rate_limit_delay)  # Delay to respect rate limits
    
    return records

def evaluate_metric(dataset, metric, judge_llm, judge_embeddings):
    name = getattr(metric, "name", metric.__class__.__name__)
    print(f"Evaluating metric: {name}")
    try:
        result = evaluate(
            dataset,
            llm=judge_llm,
            embeddings=judge_embeddings,
            metrics=[metric],
            raise_exceptions=False,
            run_config=RunConfig(
                max_workers=1,      # sequential — prevents Groq rate limit bursts
                timeout=180,        # 2 min per judge call
                max_retries=5,
                max_wait=60,
            ),
        )
        
        df = result.to_pandas()
        
        cols = [c for c in df.columns
                if c not in {"user_input", "response", "retrieved_contexts",
                             "reference", "question", "answer", "ground_truth",
                             "contexts"}]
        
        if not cols:
            print(f"Warning: No score columns found for metric {name}.")
            return name, None, None
        col = cols[0]
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        
        if valid.empty:
            print(f"  {name}: all {len(series)} evaluations failed to parse")
            return name, None, df[[col]]

        score = float(valid.mean())
        print(f"  {name}: {score:.4f}  ({len(valid)}/{len(series)} parsed)")
        return name, score, df[[col]]
            
        
    except Exception as e:
        print(f"Error evaluating metric {name}: {e}")
        return name, None, None

def build_judge(model_name: str):
    return LangchainLLMWrapper(
        ChatGroq(
            model=model_name,
            temperature=0,
            api_key=settings.groq_api_key,
            max_retries=3,
            timeout=90,
        )
    )

def run_eval():
    """Run evaluation on the golden dataset."""
    chain, retriever = build_chain()
    
    with open("data/eval/golden_dataset.json", "r") as f:
        golden_dataset = json.load(f)
    
    predictions = collect_predictions(chain, retriever, golden_dataset)
    predictions = sample_records(predictions, n=25)  # Limit to 25 questions for evaluation
    
    if(len(predictions)<len(golden_dataset)):
        print(f"Warning: Only {len(predictions)} out of {len(golden_dataset)} questions were processed. Check the checkpoint file for details.")
    
    
        
    dataset = Dataset.from_dict({
        # new-style names (RAGAS 0.2.x)
        "user_input":         [r["question"] for r in predictions],
        "response":           [r["answer"] for r in predictions],
        "retrieved_contexts": [r["context"] for r in predictions],
        "reference":          [r["ground_truth"] for r in predictions],
        # legacy aliases — harmless duplicates, cover both mapping paths
        "question":           [r["question"] for r in predictions],
        "answer":             [r["answer"] for r in predictions],
        "ground_truth":       [r["ground_truth"] for r in predictions],
    })
    
   
    '''
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model=judge_model,
            temperature=0,
            api_key=settings.groq_api_key,
            max_retries = 3,
            timeout = 90
        )
    )'''
    
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    ))
    
    scores = {}
    per_question = pd.DataFrame({"question": [r["question"] for r in predictions]})
    
    metric_cache = load_metric_cache()
    
    for metric in metrics:
        name = getattr(metric, "name", metric.__class__.__name__)
        if name in metric_cache and metric_cache[name] is not None:
            print(f"Using cached score for metric: {name}")
            scores[name] = metric_cache[name]
            continue
        
        try:
            model_name = JUDGE_BY_METRIC.get(name, "openai/gpt-oss-20b")
            print(f"\n--- {name} (judge: {model_name}) ---")
            judge_llm = build_judge(model_name)
            
            name,score,col_df = evaluate_metric(dataset, metric, judge_llm, judge_embeddings)
            scores[name] = score
            metric_cache[name] = score
            save_metric_cache(metric_cache)
        except Exception as e:
            print(f"Error evaluating metric {name}: {e}")
            save_metric_cache(metric_cache)
            print(f"Partial results for metric {name} stored.")
            break
        
        if col_df is not None:
            per_question = pd.concat([per_question, col_df.reset_index(drop=True)], axis=1)
        time.sleep(10)  # Delay between metrics to avoid rate limits
    
    print("\n" + "=" * 45)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 45)
    for name, score in scores.items():
        display = f"{score:.4f}" if score is not None else "FAILED"
        print(f"  {name:25} {display}")
    print("=" * 45)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "judge_model": judge_model,
            "generation_model": settings.llm_model,
            "questions_evaluated": len(predictions),
            "scores": scores,
        }, f, indent=2)

    per_question.to_csv(OUT_DIR / "per_question_results.csv", index=False)
    print(f"\nSaved to {OUT_DIR}/")

    return scores


if __name__ == "__main__":
    run_eval()
    

