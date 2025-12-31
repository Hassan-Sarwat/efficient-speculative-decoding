
import logging
from typing import Optional, Dict, List
from datasets import load_dataset, load_from_disk
from pathlib import Path

logger = logging.getLogger(__name__)

def load_and_filter_dataset(
    dataset_name: str, 
    split: str = "train", 
    filters: Optional[Dict[str, List[str]]] = None
):
    """Loads and filters a dataset."""
    logger.info(f"Loading dataset {dataset_name}...")
    try:
        if dataset_name == "gsm8k":
            dataset = load_dataset("gsm8k", "main", split=split, streaming=False)
        elif dataset_name == "qwedsacf/competition_math" or dataset_name == "hendrycks/competition_math":
             dataset = load_dataset(dataset_name, split=split, streaming=False)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=False)
        
        # Apply filters
        if filters:
            logger.info(f"Applying filters: {filters}")
            for key, valid_values in filters.items():
                # Filter logic: Keep if sample[key] is in valid_values
                # We handle robustly if 'key' might be missing
                dataset = dataset.filter(lambda x: str(x.get(key, "")) in valid_values)
            logger.info(f"Filtered dataset size: {len(dataset)}")

        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
