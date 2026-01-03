import logging
import json
import os
from typing import Optional, Dict, List, Any
from datasets import load_dataset, Dataset  
from pathlib import Path

logger = logging.getLogger(__name__)

class DatasetLoadError(Exception):
    """Exception raised when dataset loading or filtering fails."""
    pass

def _load_from_hub(dataset_name: str, split: str) -> Dataset:
    """Load dataset from Hugging Face Hub.
    
    Args:
        dataset_name: Dataset identifier
        split: Split to load
        
    Returns:
        Raw dataset from Hub
        
    Raises:
        DatasetLoadError: If download fails
    """
    try:
        # Special case for GSM8K (requires subset name)
        if dataset_name == "gsm8k":
            return load_dataset("gsm8k", "main", split=split, streaming=False)
        
        # Standard loading for all other datasets
        return load_dataset(dataset_name, split=split, streaming=False)
        
    except Exception as e:
        raise DatasetLoadError(f"Failed to download {dataset_name}: {e}") from e

def _apply_filters(dataset: Dataset, filters: Dict[str, List[str]]) -> Dataset:
    """Apply strict filters to dataset.
    
    Args:
        dataset: Input dataset
        filters: Filter specifications
        
    Returns:
        Filtered dataset
        
    Raises:
        ValueError: If filter key doesn't exist or result is empty
    """
    logger.info(f"Applying filters: {filters}")
    
    # Get available columns
    available_columns = (
        dataset.column_names 
        if hasattr(dataset, "column_names") 
        else list(dataset.features.keys())
    )
    
    # Apply each filter
    for key, valid_values in filters.items():
        # Validate filter key exists
        if key not in available_columns:
            raise ValueError(
                f"Filter key '{key}' not found in dataset.\n"
                f"Available columns: {available_columns}"
            )
        
        # Define filter function
        def filter_fn(sample: Dict[str, Any]) -> bool:
            """Check if sample matches filter criteria."""
            if key not in sample:
                return False
            value = sample[key]
            if value is None:
                return False
            return str(value) in valid_values
        
        # Apply filter
        original_size = len(dataset)
        dataset = dataset.filter(filter_fn)
        
        kept_pct = (len(dataset) / original_size * 100) if original_size > 0 else 0
        logger.info(
            f"  Filter '{key}': {original_size:,} → {len(dataset):,} samples "
            f"(kept {kept_pct:.1f}%)"
        )
    
    # Check for empty result
    if len(dataset) == 0:
        raise ValueError(
            f"All samples filtered out!\n"
            f"Filters: {filters}\n"
            f"No samples match your criteria."
        )
    
    return dataset

def load_and_filter_dataset(
    dataset_name: str, 
    split: str = "train", 
    filters: Optional[Dict[str, List[str]]] = None,
    output_dir: str = "data/raw",
    safe_name: str = "dataset",
    suffix: Optional[str] = None,
    limit: Optional[int] = None
) -> Dataset:
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
        limit: Optional limit on the number of samples to load/save.

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
            logger.info(f"Loaded {len(dataset):,} samples from cache")
            
            # Additional check: if limit was requested but cache has more/less, we just use cache?
            # User wants "passed to chain of thought", assuming cache IS the source of truth for "this run config"
            # So we just return what's in the cache. 
            # If user changes limit, they likely need a new suffix or delete cache, as warned in plan.
            
            return dataset
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.warning(f"Cache load failed: {e}. Re-downloading...")
        except Exception as e:
            logger.error(f"Unexpected error loading cache: {e}")
            raise DatasetLoadError(f"Cache corrupted: {e}") from e

    logger.info(f"Loading dataset {dataset_name} (split={split})...")
    try:
        # 2. Load the dataset from Hub
        dataset = _load_from_hub(dataset_name, split)
        
        # 3. Apply Strict Filters
        if filters:
            dataset = _apply_filters(dataset, filters)
        
        # 4. Apply Limit (Slice)
        if limit is not None:
            original_len = len(dataset)
            dataset = dataset.select(range(min(len(dataset), limit)))
            logger.info(f"Applied limit {limit}: {original_len:,} → {len(dataset):,} samples")

        logger.info(f"Final dataset size: {len(dataset)}")
        
        # 5. Save to local cache
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching dataset to {local_path}...")
            dataset.to_json(local_path)
            
            cache_size = local_path.stat().st_size
            logger.info(f"Cache saved ({cache_size:,} bytes)")
            
        except (IOError, OSError) as e:
            # Non-fatal - dataset is still loaded
            logger.warning(f"Failed to save cache: {e}")
        except Exception as e:
            logger.error(f"Unexpected cache error: {e}")
        # Don't fail entire operation for cache issues

        return dataset

    except DatasetLoadError:
        # Already logged in helper functions
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {dataset_name}: {e}")
        raise DatasetLoadError(f"Dataset loading failed: {e}") from e