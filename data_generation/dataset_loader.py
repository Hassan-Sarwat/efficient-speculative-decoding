import logging
import json
import os
from typing import Optional, Dict, List, Any
from datasets import load_dataset, load_from_disk
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_filter_dataset(
    dataset_name: str, 
    split: str = "train", 
    filters: Optional[Dict[str, List[str]]] = None,
    output_dir: str = "data",
    safe_name: str = "dataset",
    suffix: Optional[str] = None
):
    """
    Loads and filters a dataset with strict schema validation.
    Checks for local cached version first.
    
    Args:
        dataset_name: Name of the dataset to load.
        split: Dataset split to load.
        filters: Dictionary of filters to apply.
        output_dir: Directory to save/load cached dataset.
        safe_name: Safe filename version of dataset name.
        suffix: Optional suffix for the dataset filename.

    Raises:
        ValueError: If a filter key is missing from the dataset or if filtering results in empty data.
    """
    
    # Determine local file path
    if suffix:
        filename = f"{safe_name}_{suffix}.jsonl"
    else:
        filename = f"{safe_name}.jsonl"
    
    local_path = Path(output_dir) / filename
    
    # 1. Check for local existence
    if local_path.exists():
        logger.info(f"Found local cached dataset at {local_path}. Loading...")
        try:
            dataset = load_dataset("json", data_files=str(local_path), split="train")
            logger.info(f"Loaded {len(dataset)} samples from local file.")
            return dataset
        except Exception as e:
            logger.warning(f"Failed to load local file {local_path}: {e}. Falling back to download.")

    logger.info(f"Loading dataset {dataset_name} (split={split})...")
    try:
        # 2. Load the dataset from Hub
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split, streaming=False)
        elif dataset_name in ["qwedsacf/competition_math", "hendrycks/competition_math"]:
             dataset = load_dataset(dataset_name, split=split, streaming=False)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        # 3. Apply Strict Filters
        if filters:
            logger.info(f"Applying filters: {filters}")
            
            # Get available columns
            available_columns = dataset.column_names if hasattr(dataset, "column_names") else dataset.features.keys()
            
            for key, valid_values in filters.items():
                if key not in available_columns:
                    raise ValueError(
                        f"Filter key '{key}' not found in dataset columns: {available_columns}. "
                        "Check your spelling or dataset version."
                    )
                
                def strict_filter(sample: Dict[str, Any]) -> bool:
                    if key not in sample:
                        return False
                    val = sample[key]
                    if val is None:
                        return False
                    return str(val) in valid_values

                original_size = len(dataset)
                dataset = dataset.filter(strict_filter)
                logger.info(f"Filtered '{key}': {original_size} -> {len(dataset)} samples")

            if len(dataset) == 0:
                raise ValueError(
                    f"CRITICAL: Dataset is empty after filtering! "
                    f"Filters applied: {filters}. "
                )

        logger.info(f"Final dataset size: {len(dataset)}")
        
        # 4. Save to local cache
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving filtered dataset to {local_path}...")
            dataset.to_json(local_path)
            logger.info("Save complete.")
        except Exception as e:
            logger.warning(f"Failed to save local cache to {local_path}: {e}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise