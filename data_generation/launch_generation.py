
import os
import json
import logging
import argparse
import time
import sys
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import uuid

# Add project root to sys.path to allow imports from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompts import SAFETY_SETTINGS
from batch_client import BatchClient
from dataset_loader import load_and_filter_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("launch_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_existing_questions(output_file: Path) -> set:
    """Retrieves questions that have already been generated."""
    existing = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "instruction" in data:
                        existing.add(data["instruction"])
                except json.JSONDecodeError:
                    continue
    return existing

def get_all_existing_questions(output_dir: Path, safe_name: str) -> set:
    """Retrieves questions that have already been generated across all matching files."""
    existing = set()
    # Match all files starting with cob_ and ending with .jsonl to cover all variatons
    for file_path in output_dir.glob("cob_*.jsonl"):
        existing.update(get_existing_questions(file_path))
    return existing

def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    parser = argparse.ArgumentParser(description="Launch Chain of Draft Generation Batch")
    parser.add_argument("--model", type=str, default="gemini-3.0-pro-preview", help="Model ID to use")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset name to use")
    parser.add_argument("--limit", type=int, default=1000, help="Number of NEW samples to generate in this run")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save final output")
    parser.add_argument("--temp_dir", type=str, default="tmp", help="Directory for temporary batch files")
    parser.add_argument("--file_suffix", type=str, help="Custom suffix for output file (e.g. 'medium' -> cob_medium.jsonl)")
    parser.add_argument("--filter", action='append', help="Filter dataset. Format: key=val1,val2 (e.g. type=Algebra)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare batch file but do not submit")
    args = parser.parse_args()

    # Parse filters
    filters = {}
    if args.filter:
        for f in args.filter:
            try:
                key, vals = f.split("=", 1)
                filters[key] = vals.split(",")
            except ValueError:
                logger.warning(f"Invalid filter format: {f}. Use key=val1,val2")

    # Ensure directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    # File paths
    safe_name = args.dataset.replace("/", "_")
    
    # Determine output file name
    if args.file_suffix:
        base_name = f"cob_{args.file_suffix}"
    else:
        base_name = f"cob_data_{safe_name}"
        
    output_file = Path(args.output_dir) / f"{base_name}.jsonl"
    
    count_file = 0
    # If the base file exists, we start looking for indexed files
    if output_file.exists():
        count_file = 1
        while (Path(args.output_dir) / f"{base_name}_{count_file}.jsonl").exists():
            count_file += 1
        output_file = Path(args.output_dir) / f"{base_name}_{count_file}.jsonl"
    
    batch_input_file = Path(args.temp_dir) / f"batch_input_gen_{safe_name}_{int(time.time())}.jsonl"
    
    # State file also gets the suffix if it exists, to separate concerns
    if args.file_suffix:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}_{args.file_suffix}.json"
    else:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}.json"

    # Load Dataset
    dataset = load_and_filter_dataset(args.dataset, split="train", filters=filters)
    
    # Check existing across ALL files to avoid duplicates
    existing = get_all_existing_questions(Path(args.output_dir), safe_name)
    logger.info(f"Found {len(existing)} existing samples across all cob_*.jsonl files.")
    logger.info(f"Writing new samples to: {output_file}")

    # Prepare requests
    requests = []
    count = 0
    needed = args.limit
    
    logger.info(f"Targeting generation of {needed} new samples.")

    mapping = {}

    for sample in tqdm(dataset, desc="Processing dataset"):
        if count >= needed:
            break
        
        question = sample.get("question") or sample.get("prompt") or sample.get("input") or sample.get("instruction")
        problem = sample.get("problem")
        if not question and problem:
            question = problem

        if not question:
            continue
            
        if question in existing:
            continue
            
        request_body = {
            "contents": [{"parts": [{"text": question}]}],
            "generationConfig": {
                "thinking_config": {
                    "include_thoughts": True,
                    "thinking_level": "HIGH"
                }
            },
            "safetySettings": SAFETY_SETTINGS
        }
        
        custom_id = f"req_{uuid.uuid4().hex[:12]}"  
        requests.append({
            "custom_id": custom_id,
            "request": request_body
        })
        mapping[custom_id] = question
        count += 1

    if not requests:
        logger.info("No new requests generated.")
        return

    # Write batch file
    with open(batch_input_file, "w", encoding="utf-8") as f:
        for r in requests:
            line_obj = {"custom_id": r["custom_id"], "request": r["request"]}
            f.write(json.dumps(line_obj) + "\n")
    
    logger.info(f"Batch file created: {batch_input_file} with {len(requests)} requests.")

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # Submit Batch
    client = BatchClient(api_key=api_key)
    batch_name = client.submit_batch(batch_input_file, args.model)
    
    # Update State
    state = {}
    if batch_state_file.exists():
        try:
            with open(batch_state_file, "r") as f:
                state = json.load(f)
        except:
            logger.warning(f"Could not load state: {e}. Starting fresh.")
            state = {}
    
    # We append the new batch logic to state, keeping history could be useful
    if "batches" not in state:
        state["batches"] = []
        
    state["batches"].append({
        "batch_name": batch_name,
        "type": "generation",
        "mapping": mapping,
        "status": "submitted",
        "timestamp": time.time(),
        "model": args.model,
        "output_file": str(output_file) # Save specific output file for this batch
    })
    
    with open(batch_state_file, "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Batch submitted: {batch_name}")
    logger.info(f"State saved to {batch_state_file}")
    logger.info("Run 'python utils/process_results.py' to check status and download results.")

if __name__ == "__main__":
    main()
