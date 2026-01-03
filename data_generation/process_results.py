import os
import json
import logging
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import asdict

from prompts import SUMMARIZATION_PROMPT
from batch_client import BatchClient
from result_processor import (
    ProcessingMetrics,
    TrainingSample,         
    IntermediateSample,      
    process_generation_results, 
    process_summarization_results, 
    get_existing_instructions
)

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

def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
        
    parser = argparse.ArgumentParser(description="Process Chain of Draft Batch Results")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", 
                       help="Dataset name to identify state file")
    parser.add_argument("--temp_dir", type=str, default="tmp", 
                       help="Directory for state files")
    parser.add_argument("--output_dir", type=str, default="data", 
                       help="Directory for final output")
    parser.add_argument("--file_suffix", type=str, help="Custom suffix to identify state file")
    parser.add_argument("--overwrite_metrics", action="store_true", help="Overwrite existing metrics file instead of appending")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    
    if args.file_suffix:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}_{args.file_suffix}.json"
        metrics_file = Path(args.output_dir) / f"metrics_{safe_name}_{args.file_suffix}.json"
    else:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}.json"
        metrics_file = Path(args.output_dir) / f"metrics_{safe_name}.json"
    
    if not batch_state_file.exists():
        logger.error(f"State file {batch_state_file} not found.")
        return

    with open(batch_state_file, "r") as f:
        state = json.load(f)

    if "batches" not in state:
        logger.info("No batches found in state.")
        return

    # Load or initialize metrics
    if metrics_file.exists() and not args.overwrite_metrics:
        logger.info(f"Loading existing metrics from {metrics_file}")
        metrics = ProcessingMetrics.load_from_file(metrics_file)
    else:
        if args.overwrite_metrics and metrics_file.exists():
            logger.info(f"Overwriting existing metrics file: {metrics_file}")
        metrics = ProcessingMetrics()
    
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
            results_path = client.download_results(job, Path(args.temp_dir))
            if not results_path:
                updated_batches.append(batch)
                continue
                
            if batch["type"] == "generation":
                target_output_file = Path(batch.get("output_file", Path(args.output_dir) / f"cob_data_{safe_name}.jsonl"))
                
                # Derive CoT and CoD filenames
                filename = target_output_file.name
                if filename.startswith("cot_"):
                   cot_filename = filename
                   cod_filename = filename.replace("cot_", "cob_", 1)
                elif filename.startswith("cob_"): 
                   cod_filename = filename 
                   cot_filename = filename.replace("cob_", "cot_", 1)
                else:
                   cod_filename = f"cob_{filename}"
                   cot_filename = f"cot_{filename}"
                   
                cod_file = target_output_file.parent / cod_filename
                cot_file = target_output_file.parent / cot_filename

                cod_ready, cot_ready, to_summarize = process_generation_results(
                    results_path, 
                    batch["mapping"],
                    metrics 
                )
                
                # Save CoT samples
                if cot_ready:
                    existing_cot = get_existing_instructions(cot_file)
                    new_cot = [item for item in cot_ready if item.instruction not in existing_cot]  
    
                    if new_cot:
                        with open(cot_file, "a", encoding="utf-8") as f:
                            for item in new_cot:
                                f.write(json.dumps(asdict(item)) + "\n")
                        logger.info(f"Saved {len(new_cot)} CoT samples to {cot_file}. (Skipped {len(cot_ready) - len(new_cot)} duplicates)")

                # Save ready CoD samples
                if cod_ready:
                    existing_cod = get_existing_instructions(cod_file)
                    new_cod = [item for item in cod_ready if item.instruction not in existing_cod]  
    
                    if new_cod:
                        with open(cod_file, "a", encoding="utf-8") as f:
                            for item in new_cod:
                                f.write(json.dumps(asdict(item)) + "\n")
                        logger.info(f"Saved {len(new_cod)} CoD samples to {cod_file}. (Skipped {len(cod_ready) - len(new_cod)} duplicates)")

                # Submit summarization if needed
                if to_summarize:
                    logger.info(f"Submitting summarization for {len(to_summarize)} samples.")
                    
                    sum_requests = []
                    sum_mapping = {}
                    
                    for i, sample in enumerate(to_summarize):
                        req_id = f"sum_{batch_name}_{i}"
                        summary_prompt = SUMMARIZATION_PROMPT.format(question=sample.question, raw_logic=sample.raw_logic)  

                        
                        request_body = {
                            "contents": [{"parts": [{"text": summary_prompt}]}]
                        }
                        
                        sum_requests.append({
                            "key": req_id,
                            "request": request_body
                        })
                        sum_mapping[req_id] = sample
                    
                    safe_batch_name = batch_name.replace("/", "_")
                    sum_file = Path(args.temp_dir) / f"batch_input_sum_{safe_batch_name}.jsonl"
                    with open(sum_file, "w", encoding="utf-8") as f:
                        for r in sum_requests:
                            line_obj = {"key": r["key"], "request": r["request"]}
                            f.write(json.dumps(line_obj) + "\n")
                    
                    try:
                        sum_batch_name = client.submit_batch(sum_file, batch["model"])
                        updated_batches.append({
                            "batch_name": sum_batch_name,
                            "type": "summarization",
                            "mapping": {k: asdict(v) for k, v in sum_mapping.items()},  # Convert to dict
                            "status": "submitted",
                            "timestamp": time.time(),
                            "model": batch["model"],
                            "output_file": str(target_output_file)
                        })
                        batch["status"] = "done"
                        updated_batches.append(batch)
                        
                    except Exception as e:
                        logger.error(f"Failed to submit summary batch: {e}")
                        updated_batches.append(batch)
                else:
                    batch["status"] = "done"
                    updated_batches.append(batch)
            
            elif batch["type"] == "summarization":
                target_output_file = Path(batch.get("output_file", Path(args.output_dir) / f"cob_data_{safe_name}.jsonl"))
                filename = target_output_file.name
                if filename.startswith("cob_"):
                   cod_filename = filename
                elif filename.startswith("cot_"):
                   cod_filename = filename.replace("cot_", "cob_", 1)
                else:
                   cod_filename = f"cob_{filename}"
                cod_file = target_output_file.parent / cod_filename

                mapping_with_samples = {
                    key: IntermediateSample(**value)  # Convert dict to IntermediateSample
                    for key, value in batch["mapping"].items()
                }
    
                final_samples = process_summarization_results(
                    results_path, 
                    mapping_with_samples,  # Now correct type
                    metrics 
                )

                
                if final_samples:
                    existing_cod = get_existing_instructions(cod_file)
                    new_samples = [item for item in final_samples if item.instruction not in existing_cod]
                    
                    if new_samples:
                        with open(cod_file, "a", encoding="utf-8") as f:
                            for item in new_samples:
                                f.write(json.dumps(asdict(item)) + "\n")
                        logger.info(f"Saved {len(new_samples)} summarized samples to {cod_file}. (Skipped {len(final_samples) - len(new_samples)} duplicates)")
                
                batch["status"] = "done"
                updated_batches.append(batch)
        
        else:
            updated_batches.append(batch)

    state["batches"] = updated_batches
    with open(batch_state_file, "w") as f:
        json.dump(state, f, indent=2)
    
    print(metrics.get_summary())
    metrics.save_to_file(metrics_file)

if __name__ == "__main__":
    main()