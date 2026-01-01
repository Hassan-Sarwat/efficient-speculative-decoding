import logging
from typing import Optional, Dict, List, Any
from datasets import load_dataset, load_from_disk
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_filter_dataset(
    dataset_name: str, 
    split: str = "train", 
    filters: Optional[Dict[str, List[str]]] = None
):
    """
    Loads and filters a dataset with strict schema validation.
    
    Raises:
        ValueError: If a filter key is missing from the dataset or if filtering results in empty data.
    """
    logger.info(f"Loading dataset {dataset_name} (split={split})...")
    try:
        # 1. Load the dataset
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split, streaming=False)
        elif dataset_name in ["qwedsacf/competition_math", "hendrycks/competition_math"]:
             dataset = load_dataset(dataset_name, split=split, streaming=False)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        # 2. Apply Strict Filters
        if filters:
            logger.info(f"Applying filters: {filters}")
            
            # Get available columns to prevent "silent" failures on wrong keys
            available_columns = dataset.column_names if hasattr(dataset, "column_names") else dataset.features.keys()
            
            for key, valid_values in filters.items():
                # Safety Check 1: Does the column exist?
                if key not in available_columns:
                    raise ValueError(
                        f"Filter key '{key}' not found in dataset columns: {available_columns}. "
                        "Check your spelling or dataset version."
                    )
                
                # Define a strict filter function (no lambdas)
                def strict_filter(sample: Dict[str, Any]) -> bool:
                    if key not in sample:
                        return False
                    
                    val = sample[key]
                    
                    # Handle None values safely
                    if val is None:
                        return False
                        
                    # Handle Type Mismatch (CLI args are always strings)
                    # If dataset has Int(1) and filter is Str("1"), we match.
                    # But we do NOT match "Level 1" to "1" blindly.
                    return str(val) in valid_values

                original_size = len(dataset)
                dataset = dataset.filter(strict_filter)
                logger.info(f"Filtered '{key}': {original_size} -> {len(dataset)} samples")

            # Safety Check 2: Did we lose everything?
            if len(dataset) == 0:
                raise ValueError(
                    f"CRITICAL: Dataset is empty after filtering! "
                    f"Filters applied: {filters}. "
                    "Double check that your filter values match the dataset format (e.g., '1' vs 'Level 1')."
                )

        logger.info(f"Final dataset size: {len(dataset)}")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise