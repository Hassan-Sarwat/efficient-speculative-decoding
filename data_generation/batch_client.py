
import logging
from typing import Any, Optional
from google import genai
from pathlib import Path

logger = logging.getLogger(__name__)

class BatchClient:
    """Wrapper for Google Gemini Batch API."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def submit_batch(self, file_path: Path, model_id: str) -> str:
        """Uploads a file and creates a batch job."""
        logger.info(f"Uploading file {file_path}...")
        try:
            batch_file = self.client.files.upload(file=str(file_path), config={'mime_type': 'application/jsonl'})
            logger.info(f"File uploaded: {batch_file.name}")
            
            logger.info(f"Creating batch job for model {model_id}...")
            batch_job = self.client.batches.create(
                model=model_id,
                src=batch_file.name,
            )
            
            logger.info(f"Batch job created: {batch_job.name}")
            return batch_job.name
            
        except Exception as e:
            logger.error(f"Error submitting batch: {e}")
            raise

    def check_batch_status(self, batch_name: str) -> Any:
        """Checks the status of a running batch job."""
        logger.info(f"Checking status for {batch_name}...")
        try:
            batch_job = self.client.batches.get(name=batch_name)
            return batch_job
        except Exception as e:
            logger.error(f"Error checking status: {e}")
            return None

    def download_results(self, batch_job: Any) -> Optional[str]:
        """Downloads the results of a completed batch job."""
        logger.info("Downloading results...")
        try:
            output_name = None
            if hasattr(batch_job, 'dest') and batch_job.dest:
                output_name = getattr(batch_job.dest, 'name', None)
                if not output_name:
                     output_name = getattr(batch_job.dest, 'file_name', None)
            
            if not output_name:
                logger.error(f"No output file name found in batch job destination: {batch_job.dest}")
                return None
                
            logger.info(f"Downloading from: {output_name}")
            file_content = self.client.files.download(file=output_name)
            return file_content.decode('utf-8')

        except Exception as e:
            logger.error(f"Error downloading results: {e}")
            return None
