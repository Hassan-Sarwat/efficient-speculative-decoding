
import os
import json
import logging
import argparse
import time
import sys
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generation.prompts import SUMMARIZATION_PROMPT
from data_generation.batch_client import BatchClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("process_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_generation_results(results_text: str, mapping: Dict[str, str]) -> tuple[List[Dict], List[Dict]]:
    """Parses generation results into processed samples and those needing summarization."""
    lines = results_text.strip().split("\n")
    processed_samples = []
    to_summarize = []
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result = json.loads(line)
            custom_id = result.get("custom_id")
            
            if not custom_id or custom_id not in mapping:
                continue
                
            question = mapping[custom_id]
            response = result.get("response", {})
            
            if "candidates" not in response:
                if "error" in response:
                    logger.error(f"Error in generation for {custom_id}: {response['error']}")
                continue

            candidate = response["candidates"][0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            raw_logic = ""
            final_answer = ""
            
            for part in parts:
                if part.get("thought"):
                    raw_logic += part.get("text", "")
                else:
                    final_answer += part.get("text", "")
            
            sample_data = {
                "question": question,
                "raw_logic": raw_logic,
                "final_answer": final_answer,
                "custom_id": custom_id 
            }
            
            # Logic to determine if summarization is needed
            if len(raw_logic.split()) > 100:
                to_summarize.append(sample_data)
            else:
                if not raw_logic:
                    logger.warning(f"Skipping sample {custom_id} due to missing reasoning.")
                    continue

                formatted_output = f"<draft> {raw_logic} </draft> #### {final_answer}"
                processed_samples.append({
                    "instruction": question,
                    "input": "",
                    "output": formatted_output
                })
                
        except Exception as e:
            logger.error(f"Error processing line {i}: {e}")
            continue
            
    return processed_samples, to_summarize

def process_summarization_results(results_text: str, mapping: Dict[str, Dict]) -> List[Dict]:
    """Parses summarization results and merges them with original samples."""
    lines = results_text.strip().split("\n")
    final_samples = []
    
    for i, line in enumerate(lines):
        try:
            result = json.loads(line)
            custom_id = result.get("custom_id")
            
            if not custom_id or custom_id not in mapping:
                continue

            response = result.get("response", {})
            if "candidates" not in response:
                continue

            candidate = response["candidates"][0]
            summary = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])).strip()
            
            # Ensure <draft> tags format
            if not summary.startswith("<draft>"):
                summary = f"<draft> {summary}"
            if not summary.endswith("</draft>"):
                summary = f"{summary} </draft>"
            
            original_sample = mapping[custom_id]
            formatted_output = f"{summary} #### {original_sample['final_answer']}"
            
            final_samples.append({
                "instruction": original_sample["question"],
                "input": "",
                "output": formatted_output
            })
            
        except Exception as e:
            logger.error(f"Error processing summary line {i}: {e}")
            continue
            
    return final_samples

def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
        
    parser = argparse.ArgumentParser(description="Process Chain of Draft Batch Results")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset name to identify state file")
    parser.add_argument("--temp_dir", type=str, default="tmp", help="Directory for state files")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory for final output")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}.json"
    output_file = Path(args.output_dir) / f"cob_data_{safe_name}.jsonl"
    
    if not batch_state_file.exists():
        logger.error(f"State file {batch_state_file} not found.")
        return

    with open(batch_state_file, "r") as f:
        state = json.load(f)

    if "batches" not in state:
        logger.info("No batches found in state.")
        return

    client = BatchClient(api_key=api_key)
    
    updated_batches = []
    
    for batch in state["batches"]:
        if batch["status"] == "done":
            updated_batches.append(batch)
            continue
            
        batch_name = batch["batch_name"]
        job = client.check_batch_status(batch_name)
        
        logger.info(f"Batch {batch_name} ({batch['type']}): {job.state}")
        
        if str(job.state) == "JobState.JOB_STATE_SUCCEEDED":
            results_text = client.download_results(job)
            if not results_text:
                updated_batches.append(batch)
                continue
                
            if batch["type"] == "generation":
                processed, to_summarize = process_generation_results(results_text, batch["mapping"])
                
                # Save processed directly
                if processed:
                     with open(output_file, "a", encoding="utf-8") as f:
                        for item in processed:
                            f.write(json.dumps(item) + "\n")
                     logger.info(f"Saved {len(processed)} samples.")

                # Submit summarization if needed
                if to_summarize:
                    logger.info(f"Submitting summarization for {len(to_summarize)} samples.")
                    
                    sum_requests = []
                    sum_mapping = {}
                    
                    for i, sample in enumerate(to_summarize):
                        req_id = f"sum_{batch_name}_{i}" # Unique ID derived from parent batch
                        
                        summary_prompt = SUMMARIZATION_PROMPT.format(raw_logic=sample['raw_logic'])
                        
                        request_body = {
                            "contents": [{"parts": [{"text": summary_prompt}]}]
                        }
                        
                        sum_requests.append({
                            "custom_id": req_id,
                            "request": request_body
                        })
                        sum_mapping[req_id] = sample
                    
                    sum_file = Path(args.temp_dir) / f"batch_input_sum_{batch_name}.jsonl"
                    with open(sum_file, "w", encoding="utf-8") as f:
                        for r in sum_requests:
                            line_obj = {"custom_id": r["custom_id"], "request": r["request"]}
                            f.write(json.dumps(line_obj) + "\n")
                    
                    try:
                        sum_batch_name = client.submit_batch(sum_file, batch["model"]) # Reuse same model or config? Usually small model is fine but consistent is ok.
                        updated_batches.append({
                            "batch_name": sum_batch_name,
                            "type": "summarization",
                            "mapping": sum_mapping,
                            "status": "submitted",
                            "timestamp": time.time(),
                            "model": batch["model"]
                        })
                        batch["status"] = "done" # Mark parent generation as done
                        updated_batches.append(batch)
                        
                    except Exception as e:
                        logger.error(f"Failed to submit summary batch: {e}")
                        updated_batches.append(batch) # Keep it to retry or manual check?
                else:
                    batch["status"] = "done"
                    updated_batches.append(batch)
            
            elif batch["type"] == "summarization":
                final_samples = process_summarization_results(results_text, batch["mapping"])
                if final_samples:
                    with open(output_file, "a", encoding="utf-8") as f:
                        for item in final_samples:
                            f.write(json.dumps(item) + "\n")
                    logger.info(f"Saved {len(final_samples)} summarized samples.")
                
                batch["status"] = "done"
                updated_batches.append(batch)
        
        else:
            updated_batches.append(batch) # Keep checking

    state["batches"] = updated_batches
    with open(batch_state_file, "w") as f:
        json.dump(state, f, indent=2)

if __name__ == "__main__":
    main()
