import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Set, Tuple, List
from dotenv import load_dotenv
from dataclasses import dataclass, asdict

from batch_client import BatchClient

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

@dataclass
class TrainingSample:
    """Standard training sample format for fine-tuning."""
    instruction: str
    input: str
    output: str
    id: str | None = None

def get_existing_instructions(filepath: Path) -> Set[str]:
    """Reads a JSONL file and returns a set of existing instructions (questions) to avoid duplicates."""
    existing = set()
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            existing.add(data["instruction"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error reading existing file {filepath}: {e}")
    return existing

def parse_generated_content(response: Dict[str, Any]) -> str:
    """
    Extracts the text content from a Gemini candidate object.
    Concatenates all parts.
    """
    if "body" in response:
        response = response["body"]
        
    candidates = response.get("candidates", [])
    if not candidates:
        return ""
        
    candidate = candidates[0]
    content = candidate.get("content", {})
    parts = content.get("parts", [])
    
    full_text = ""
    for part in parts:
        full_text += part.get("text", "")
        
    return full_text.strip()

def process_batch_row(
    result: Dict[str, Any], 
    mapping: Dict[str, Any]
) -> List[TrainingSample]:
    """
    Process a single row from the batch output.
    Returns a list of samples (usually 1, or 0 if error).
    """
    custom_id = result.get("key")
    if not custom_id or custom_id not in mapping:
        return []
    
    # Handle both new (dict) and old (str) mapping formats
    map_entry = mapping[custom_id]
    if isinstance(map_entry, dict):
        question = map_entry.get("question")
        sample_id = map_entry.get("id")
    else:
        question = map_entry
        sample_id = None

    raw_response = result.get("response", {})
    
    # helper handles the 'body' nesting if present
    output_text = parse_generated_content(raw_response)
    
    if not output_text:
        return []

    # Create the sample
    sample = TrainingSample(
        instruction=question,
        input="",
        output=output_text,
        id=sample_id
    )
    
    return [sample]


def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
        
    parser = argparse.ArgumentParser(description="Process Chain of Draft Batch Results")
    parser.add_argument("--chain", type=str, choices=['thought', 'draft'], default='thought', help="Type of chain: 'thought' for CoT, 'draft' for CoD.")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", 
                       help="Dataset name to identify state file")
    parser.add_argument("--temp_dir", type=str, default="tmp", 
                       help="Directory for state files")
    parser.add_argument("--output_dir", type=str, default="data", 
                       help="Directory for final output")
    parser.add_argument("--file_suffix", type=str, help="Custom suffix to identify state file")
    parser.add_argument("--force", action="store_true", help="Force re-check and re-download of batches marked as done")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    
    # Determine base prefix based on chain
    if args.chain == 'thought':
        prefix = 'cot'
    else:
        prefix = 'cod'

    if args.file_suffix:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{prefix}_{safe_name}_{args.file_suffix}.json"
    else:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{prefix}_{safe_name}.json"
    
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
        if batch["status"] == "done" and not args.force:
            updated_batches.append(batch)
            continue
            
        batch_name = batch["batch_name"]
        try:
            job = client.check_batch_status(batch_name)
            logger.info(f"Batch {batch_name} ({batch['type']}): {job.state}")
            
            if str(job.state) == "JobState.JOB_STATE_SUCCEEDED":
                results_path = client.download_results(job, Path(args.temp_dir))
                
                if results_path and results_path.stat().st_size > 0:
                    # Determine target file
                    if args.chain == 'thought':
                        target_output_file = Path(batch.get("output_file", Path(args.output_dir) / "training" / f"cot_{safe_name}.jsonl"))
                    else: 
                        target_output_file = Path(batch.get("output_file", Path(args.output_dir) / "training" / f"cod_{safe_name}.jsonl"))
                    
                    mapping = batch["mapping"]
                    new_samples = []

                    # Process the results file
                    with open(results_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                result = json.loads(line)
                                samples = process_batch_row(result, mapping)
                                new_samples.extend(samples)
                            except json.JSONDecodeError:
                                logger.error("JSON decode error in results file")
                                continue

                    # Save to final output
                    if new_samples:
                        existing_items = get_existing_instructions(target_output_file)
                        unique_new_items = [item for item in new_samples if item.instruction not in existing_items]  
        
                        if unique_new_items:
                            with open(target_output_file, "a", encoding="utf-8") as output_f:
                                for item in unique_new_items:
                                    output_f.write(json.dumps(asdict(item)) + "\n")
                            logger.info(f"Saved {len(unique_new_items)} samples to {target_output_file}. (Skipped {len(new_samples) - len(unique_new_items)} duplicates)")
                    
                    batch["status"] = "done"
                else:
                    logger.warning(f"Results file missing or empty for {batch_name}")
            
            updated_batches.append(batch)
            
        except Exception as e:
            logger.error(f"Error checking/processing batch {batch_name}: {e}")
            updated_batches.append(batch)

    state["batches"] = updated_batches
    with open(batch_state_file, "w") as f:
        json.dump(state, f, indent=2)

if __name__ == "__main__":
    main()