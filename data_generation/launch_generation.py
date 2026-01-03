
import os
import json
import logging
import argparse
import time
from typing import Dict, List, Any, Set, Optional
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import uuid
from enum import Enum

from prompts import COT_SYSTEM_INSTRUCTIONS, COD_SYSTEM_INSTRUCTIONS
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

def get_existing_questions(output_file: Path) -> Set[str]:
    """Retrieves questions that have already been generated.
    
    Args:
        output_file: Path to JSONL file
        
    Returns:
        Set of question strings found in file
    """
    existing: Set[str] = set()
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

def get_all_existing_questions(
    output_dir: Path, 
    safe_name: str, 
    suffix: Optional[str] = None,  
    prefix: str = "cot"
) -> Set[str]:
    """Retrieves questions across all matching files.
    
    Args:
        output_dir: Directory to search
        safe_name: Sanitized dataset name
        suffix: Optional file suffix
        prefix: File prefix to match
        
    Returns:
        Set of all unique questions found
    """
    existing: Set[str] = set()
    
    if suffix:
         pattern = f"{prefix}_{suffix}*.jsonl"
    else:
         pattern = f"{prefix}_{safe_name}*.jsonl"
         
    # Match all files starting with prefix and ending with .jsonl to cover all variatons
    logger.info(f"Scanning for existing questions in {output_dir} with pattern: {pattern}")
    for file_path in output_dir.glob(pattern):
        existing.update(get_existing_questions(file_path))
    return existing


def determine_output_filename(
    output_dir: Path,
    prefix: str,
    safe_name: str,
    suffix: str = None
) -> Path:
    """Determine next available output filename.
    
    If base file exists, finds next indexed filename (e.g., cot_easy_1.jsonl).
    
    Args:
        output_dir: Output directory
        prefix: File prefix (e.g., 'cot', 'cod')
        safe_name: Sanitized dataset name
        suffix: Optional custom suffix
        
    Returns:
        Path to next available file (guaranteed not to exist)
        
    Examples:
        >>> determine_output_filename(Path("data"), "cot", "gsm8k", "easy")
        Path('data/cot_easy.jsonl')  # or cot_easy_1.jsonl if exists
    """
    # Determine base name
    if suffix:
        base_name = f"{prefix}_{suffix}"
    else:
        base_name = f"{prefix}_{safe_name}"
    
    # Start with base file
    output_file = output_dir / f"{base_name}.jsonl"
    
    # If it exists, find next indexed file
    if output_file.exists():
        index = 1
        while (output_dir / f"{base_name}_{index}.jsonl").exists():
            index += 1
        output_file = output_dir / f"{base_name}_{index}.jsonl"
    
    return output_file

def determine_state_filename(
    temp_dir: Path,
    safe_name: str,
    suffix: str = None,
    prefix: str = "cot"
) -> Path:
    """Determine state file path.
    
    Args:
        temp_dir: Temporary directory
        safe_name: Sanitized dataset name
        suffix: Optional custom suffix
        prefix: Chain prefix (cot/cod)
        
    Returns:
        Path to state file
    """
    if suffix:
        return temp_dir / f"batch_state_{prefix}_{safe_name}_{suffix}.json"
    else:
        return temp_dir / f"batch_state_{prefix}_{safe_name}.json"


def extract_question_from_sample(sample: Dict[str, Any]) -> Optional[str]:
    """Extract question text from dataset sample.
    
    Tries multiple common field names used across datasets.
    
    Args:
        sample: Dataset sample dictionary
        
    Returns:
        Question text or None if not found
    """
    # Try common field names in order of preference
    question = (
        sample.get("question") or 
        sample.get("prompt") or 
        sample.get("input") or 
        sample.get("instruction") or
        sample.get("problem")
    )
    
    return question if question else None

class GenerationMode(Enum):
    """Mode for handling existing samples."""
    FILL_GAP = "fill"
    EXTEND = "extend"


def determine_samples_needed(
    existing_count: int,
    target_limit: int,
    auto_fill: bool = False,
    auto_extend: bool = False
) -> int:
    """Determine how many new samples to generate.
    
    Args:
        existing_count: Number of existing samples
        target_limit: Target total samples
        auto_fill: Auto-select fill mode
        auto_extend: Auto-select extend mode
        
    Returns:
        Number of new samples to generate
    """
    # No existing samples - generate full limit
    if existing_count == 0:
        return target_limit
    
    # Auto modes
    if auto_fill:
        needed = max(0, target_limit - existing_count)
        logger.info(
            f"Auto-fill: {needed:,} new samples "
            f"(Existing: {existing_count:,}, Target: {target_limit:,})"
        )
        return needed
    
    if auto_extend:
        logger.info(
            f"Auto-extend: {target_limit:,} new samples "
            f"(Total will be {existing_count + target_limit:,})"
        )
        return target_limit
    
    # Already at or above limit
    if existing_count >= target_limit:
        logger.info(
            f"Already have {existing_count:,} samples (≥ limit {target_limit:,}). "
            f"Adding {target_limit:,} more (extend mode)."
        )
        return target_limit
    
    # Interactive prompt
    gap = target_limit - existing_count
    print(f"\n{'-'*60}")
    print(f"Found {existing_count:,} existing samples. Target: {target_limit:,}")
    print("\nWhat would you like to do?")
    print(f"  [1] Fill Gap: Generate {gap:,} samples (Total → {target_limit:,})")
    print(f"  [2] Extend: Generate {target_limit:,} samples (Total → {existing_count + target_limit:,})")
    print(f"{'-'*60}\n")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            return gap
        elif choice == "2":
            return target_limit
        else:
            print("Invalid choice. Please enter 1 or 2.")

def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    parser = argparse.ArgumentParser(description="Launch Chain of Draft Generation Batch")
    parser.add_argument("--chain", type=str, choices=['thought', 'draft'], default='thought', help="Type of chain to generate: 'thought' for CoT, 'draft' for CoD. Defaults to 'thought'.")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Model ID to use")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset name to use")
    parser.add_argument("--limit", type=int, default=1000, help="Number of NEW samples to generate in this run")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save final output")
    parser.add_argument("--temp_dir", type=str, default="tmp", help="Directory for temporary batch files")
    parser.add_argument("--file_suffix", type=str, help="Custom suffix for output file (e.g. 'medium' -> cob_medium.jsonl)")
    parser.add_argument("--filter", action='append', help="Filter dataset. Format: key=val1,val2 (e.g. type=Algebra)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare batch file but do not submit")
    parser.add_argument("--auto_fill", action="store_true", help="Automatically select 'Fill Gap' (limit - existing) if existing data found")
    parser.add_argument("--auto_extend", action="store_true", help="Automatically select 'Extend' (add limit) if existing data found")
    
    args = parser.parse_args()

    # Determine prefix and system instructions based on chain type
    if args.chain == 'thought':
        prefix = 'cot'
        system_instruction = COT_SYSTEM_INSTRUCTIONS
    else: # draft
        prefix = 'cod'
        system_instruction = COD_SYSTEM_INSTRUCTIONS

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
    (Path(args.output_dir) / "raw").mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "training").mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    # File paths
    safe_name = args.dataset.replace("/", "_")

    # Determine output file
    output_file = determine_output_filename(
        Path(args.output_dir) / "training",
        prefix,
        safe_name,
        args.file_suffix
    )

    batch_input_file = Path(args.temp_dir) / f"batch_input_gen_{safe_name}_{int(time.time())}.jsonl"

    
    # State file also gets the suffix if it exists, to separate concerns
    # State file also gets the suffix if it exists, to separate concerns
    batch_state_file = determine_state_filename(
        Path(args.temp_dir),
        safe_name,
        args.file_suffix,
        prefix
    )

    # Load Dataset
    dataset = load_and_filter_dataset(
        args.dataset, 
        split="train", 
        filters=filters,
        output_dir=Path(args.output_dir) / "raw",
        safe_name=safe_name,
        suffix=args.file_suffix,
        limit=args.limit
    )
    
    # Check existing across ALL files to avoid duplicates
    existing = get_all_existing_questions(Path(args.output_dir) / "training", safe_name, args.file_suffix, prefix)
    logger.info(f"Found {len(existing)} existing samples across all {prefix}_*.jsonl files.")
    logger.info(f"Writing new samples to: {output_file}")

    # Determine how many samples to generate
    needed = determine_samples_needed(
        existing_count=len(existing),
        target_limit=args.limit,
        auto_fill=args.auto_fill,
        auto_extend=args.auto_extend
    )
    logger.info(f"Targeting generation of {needed} new samples.")

    mapping = {}
    requests = []  
    count = 0 
    
    # Iterate with index to use as fallback ID
    for i, sample in tqdm(enumerate(dataset), desc="Processing dataset", total=len(dataset)):
        if count >= needed:
            break
        
        question = extract_question_from_sample(sample)
        if not question or question in existing:
            continue
            
        # Extract or generate ID
        # common ID fields: 'id', 'problem_id', 'question_id'
        sample_id = (
            sample.get("id") or 
            sample.get("problem_id") or 
            sample.get("question_id") or 
            f"idx_{i}" # Fallback to dataset index
        )
        sample_id = str(sample_id) # Ensure string
        
        request_body = {
            "contents": [{"parts": [{"text": question}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]}
        }
        
        custom_id = f"req_{uuid.uuid4().hex[:12]}"  
        requests.append({
            "key": custom_id,
            "request": request_body
        })
        
        # Mapping now stores structured data including ID
        mapping[custom_id] = {
            "question": question,
            "id": sample_id
        }
        count += 1

    if not requests:
        logger.info("No new requests generated.")
        return

    # Write batch file
    with open(batch_input_file, "w", encoding="utf-8") as f:
        for r in requests:
            line_obj = {"key": r["key"], "request": r["request"]}
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
            logger.info(f"Loaded existing state from {batch_state_file}")
        except json.JSONDecodeError as e:
            logger.warning(f"State file corrupted: {e}. Starting fresh.")
            state = {}
    else:
        logger.info("No existing state file. Starting fresh.")


    
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
    logger.info("Run 'python data_generation/process_results.py' to check status and download results.")

if __name__ == "__main__":
    main()
