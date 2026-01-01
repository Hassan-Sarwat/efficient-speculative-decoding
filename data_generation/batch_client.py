
import logging
from typing import Any, Optional
from google import genai
from pathlib import Path

logger = logging.getLogger(__name__)

class BatchAPIError(Exception):
    """Exception raised for errors in the Batch API operations."""
    pass

class BatchClient:
    """Wrapper for Google Gemini Batch API."""
    
    def __init__(self, api_key: str):
        """Initialize batch client.
        
        Args:
            api_key: Google API key for authentication
            
        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        
        self.client = genai.Client(api_key=api_key)
        logger.info("BatchClient initialized")

    def submit_batch(self, file_path: Path, model_id: str) -> str:
        """Uploads a file and creates a batch job.
        
        Args:
            file_path: Path to JSONL batch input file
            model_id: Model identifier (e.g., 'gemini-3-pro-preview')
            
        Returns:
            Batch job name/ID
            
        Raises:
            FileNotFoundError: If file_path doesn't exist
            BatchAPIError: If upload or job creation fails
        """
        # Input validation
        if not file_path.exists():
            raise FileNotFoundError(f"Batch file not found: {file_path}")
        
        if not model_id:
            raise ValueError("model_id cannot be empty")
        
        logger.info(f"Uploading file {file_path}...")
        
        try:
            batch_file = self.client.files.upload(
                file=str(file_path), 
                config={'mime_type': 'application/jsonl'}
            )
            logger.info(f"File uploaded: {batch_file.name}")
            
        except Exception as e:
            # Catch file upload errors specifically
            raise BatchAPIError(f"File upload failed: {e}") from e
        
        try:
            logger.info(f"Creating batch job for model {model_id}...")
            batch_job = self.client.batches.create(
                model=model_id,
                src=batch_file.name,
            )
            
            logger.info(f"✓ Batch job created: {batch_job.name}")
            return batch_job.name
            
        except Exception as e:
            # Catch batch creation errors specifically
            raise BatchAPIError(f"Batch creation failed: {e}") from e

    def check_batch_status(self, batch_name: str) -> Optional[genai.types.BatchJob]:
        """Check status of running batch job.
        
        Args:
            batch_name: Batch job identifier
            
        Returns:
        BatchJob object if found, None if error or not found
        """
        if not batch_name:
            raise ValueError("batch_name cannot be empty")
        
        logger.info(f"Checking status for {batch_name}...")
        
        try:
            batch_job = self.client.batches.get(name=batch_name)
            logger.debug(f"Status: {batch_job.state}")
            return batch_job
            
        except Exception as e:
            # Log error but return None (non-fatal)
            logger.error(f"Error checking status for {batch_name}: {e}")
            return None

    def download_results(
        self, 
        batch_job: Any, 
        destination_dir: Path
    ) -> Path:
        """Download batch results to local file.
        
        Args:
            batch_job: Completed batch job object
            destination_dir: Directory to save results
            
        Returns:
            Path to downloaded file
            
        Raises:
            ValueError: If batch_job has no output destination
            BatchAPIError: If download fails
        """
        logger.info("Downloading batch results...")
        
        # Extract output file name
        output_name = self._extract_output_name(batch_job)
        if not output_name:
            raise ValueError(
                f"Batch job has no output destination. "
                f"Job may not be complete. State: {getattr(batch_job, 'state', 'unknown')}"
            )
        
        # Prepare local path
        destination_dir.mkdir(parents=True, exist_ok=True)
        clean_name = output_name.split("/")[-1]
        local_file_path = destination_dir / clean_name
        
        logger.info(f"Downloading {output_name} -> {local_file_path}")
        
        try:
            # Download bytes
            file_content = self.client.files.download(file=output_name)
            
            # Write to disk
            with open(local_file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"✓ Downloaded {len(file_content):,} bytes")
            return local_file_path
            
        except IOError as e:
            raise BatchAPIError(f"File write error: {e}") from e
        except Exception as e:
            raise BatchAPIError(f"Download failed: {e}") from e

    def _extract_output_name(self, batch_job: Any) -> Optional[str]:
        """Extract output file name from batch job.
        
        Args:
            batch_job: Batch job object
            
        Returns:
            Output file name or None if not found
        """
        if not hasattr(batch_job, 'dest') or not batch_job.dest:
            return None
        
        # Try different attribute names
        output_name = getattr(batch_job.dest, 'name', None)
        if not output_name:
            output_name = getattr(batch_job.dest, 'file_name', None)
        
        return output_name