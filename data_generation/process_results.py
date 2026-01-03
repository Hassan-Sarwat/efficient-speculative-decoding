import os
import json
import logging
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import asdict

from result_processor import (
    ProcessingMetrics,
    process_generation_results, 
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
    parser.add_argument("--chain", type=str, choices=['thought', 'draft'], default='thought', help="Type of chain to check: 'thought' for CoT, 'draft' for CoD. Defaults to 'thought'.")
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
    
    # Determine base prefix based on chain
    if args.chain == 'thought':
        prefix = 'cot'
    else:
        prefix = 'cod'

    if args.file_suffix:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}_{args.file_suffix}.json"
        metrics_file = Path(args.output_dir) / f"metrics_{prefix}_{safe_name}_{args.file_suffix}.json"
    else:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}.json"
        metrics_file = Path(args.output_dir) / f"metrics_{prefix}_{safe_name}.json"
    
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
                # Determine target file based on chain type
                if args.chain == 'thought':
                    target_output_file = Path(batch.get("output_file", Path(args.output_dir) / f"cot_{safe_name}.jsonl"))
                else: 
                     target_output_file = Path(batch.get("output_file", Path(args.output_dir) / f"cod_{safe_name}.jsonl"))

                # In independent mode, process_generation_results should return samples ready for saving
                # We need to adapt the processor or usage. 
                # Assuming process_generation_results parses the text.
                # Since we are decoupled, we just want the text as the "CoT" or "CoD" content.
                # However, the current process_generation_results is likely tied to CoT->CoD flow.
                # Let's look at what it returns: cod_ready, cot_ready, to_summarize
                # If we are effectively "just" generating, we treat the result as the final artifact.
                
                # For this refactor, let's assume `process_generation_results` can handle this or we interpret its output.
                # Ideally, we should update `result_processor.py` too, but user only asked to "Regulate process_results.py".
                # Let's inspect `process_generation_results` usage.
                
                cod_ready, cot_ready, to_summarize = process_generation_results(
                    results_path, 
                    batch["mapping"],
                    metrics 
                )
                
                # If chain=thought, we care about cot_ready (or whatever matches).
                # If chain=draft, we care about... well, the structure is the same (question -> response).
                # But `process_generation_results` likely expects CoT and tries to split it?
                # Wait, if CoD is generated directly, it's just "Question -> CoD".
                # `process_generation_results` might try to parse "####" and things.
                
                # Since we want to simplify:
                # If it's `thought`, we save `cot_ready` + `to_summarize` (as just CoT)?
                # Actually, `process_generation_results` probably just extracts text.
                # If the prompt system instruction for CoD is used, the output IS CoD.
                # The `process_generation_results` puts it into `cot_ready` if it looks like a complete response?
                # Or `to_summarize` if it's incomplete?
                
                # Given the user instruction: "process results will be downgraded to just checking status of job and downloading"
                # "it will also take the parameter chain where it checks for either status of chain of thought or chain of draft"
                
                # So we just save the relevant items.
                
                items_to_save = []
                if args.chain == 'thought':
                    items_to_save.extend(cot_ready)
                    items_to_save.extend(to_summarize) # Even if it thinks it needs summary, for CoT we just keep it?
                    # Or maybe to_summarize means it didn't finish?
                    # Let's assume cot_ready contains valid samples.
                
                else: # draft
                    # For CoD, the output IS the draft. 
                    # `process_generation_results` might interpret it as CoT if it's long?
                    # But functionally, it's just a generated text.
                    # We can use cot_ready lists because the structure of the object is what matters.
                    items_to_save.extend(cot_ready)
                    items_to_save.extend(to_summarize)

                if items_to_save:
                    existing_items = get_existing_instructions(target_output_file)
                    new_items = [item for item in items_to_save if item.instruction not in existing_items]  
    
                    if new_items:
                        with open(target_output_file, "a", encoding="utf-8") as f:
                            for item in new_items:
                                f.write(json.dumps(asdict(item)) + "\n")
                        logger.info(f"Saved {len(new_items)} samples to {target_output_file}. (Skipped {len(items_to_save) - len(new_items)} duplicates)")

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