import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("batch_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for the Chain of Draft generation process."""
    api_key: str
    dataset_name: str
    dataset_split: str = "train"
    model_id: str = "gemini-2.0-flash-thinking-exp-1219"
    target_total: int = 1000
    base_output_dir: str = "data"
    temp_dir: str = "tmp"
    
    # Computed properties for paths
    output_file: Path = field(init=False)
    batch_state_file: Path = field(init=False)
    batch_input_gen_file: Path = field(init=False)
    batch_input_sum_file: Path = field(init=False)
    dataset_local_path: Path = field(init=False)

    def __post_init__(self):
        # Sanitize dataset name for file paths
        safe_name = self.dataset_name.replace("/", "_")
        
        self.output_file = Path(self.base_output_dir) / f"cob_data_{safe_name}.jsonl"
        self.batch_state_file = Path(self.temp_dir) / f"batch_state_{safe_name}.json"
        self.batch_input_gen_file = Path(self.temp_dir) / f"batch_input_gen_{safe_name}.jsonl"
        self.batch_input_sum_file = Path(self.temp_dir) / f"batch_input_sum_{safe_name}.jsonl"
        self.dataset_local_path = Path(self.temp_dir) / f"{safe_name}_dataset"

class ChainOfDraftGenerator:
    """
    Manages the generation of Chain of Draft data using Google's Gemini Batch API.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self._ensure_directories()
        self.state = self._load_state()

    def _ensure_directories(self):
        """Ensures necessary directories exist."""
        Path(self.config.base_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)

    def _load_state(self) -> Dict[str, Any]:
        """Loads the batch processing state from disk."""
        if self.config.batch_state_file.exists():
            try:
                with open(self.config.batch_state_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode {self.config.batch_state_file}. Starting fresh.")
        return {}

    def _save_state(self):
        """Persists the batch processing state to disk."""
        with open(self.config.batch_state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def get_existing_questions(self) -> Set[str]:
        """Retrieves questions that have already been generated."""
        existing = set()
        if self.config.output_file.exists():
            with open(self.config.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "instruction" in data:
                            existing.add(data["instruction"])
                    except json.JSONDecodeError:
                        continue
        return existing

    def load_dataset_safe(self):
        """Loads the dataset, handling both local and remote sources."""
        if self.config.dataset_local_path.exists():
            logger.info(f"Loading dataset from local path: {self.config.dataset_local_path}")
            return load_from_disk(str(self.config.dataset_local_path))
        
        logger.info(f"Downloading dataset {self.config.dataset_name}...")
        try:
            # Add handling for specific dataset configs if needed (like gsm8k 'main')
            # For general flexibility, we try default/main first or just the name
            if self.config.dataset_name == "gsm8k":
                dataset = load_dataset("gsm8k", "main", split=self.config.dataset_split, streaming=False)
            else:
                dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split, streaming=False)
            
            dataset.save_to_disk(str(self.config.dataset_local_path))
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset {self.config.dataset_name}: {e}")
            raise

    def prepare_generation_batch(self) -> Optional[Path]:
        """Prepares a batch file for generating reasoning chains."""
        logger.info("Preparing generation batch...")
        
        dataset = self.load_dataset_safe()

        existing_questions = self.get_existing_questions()
        logger.info(f"Found {len(existing_questions)} existing samples.")
        
        needed = self.config.target_total - len(existing_questions)
        if needed <= 0:
            logger.info("Target total reached. No new samples needed.")
            return None

        logger.info(f"Need to generate {needed} samples.")
        
        requests = []
        count = 0
        
        for sample in tqdm(dataset, desc="Processing dataset"):
            if count >= needed:
                break
            
            # Flexible key lookup for different datasets
            question = sample.get("question") or sample.get("prompt") or sample.get("input") or sample.get("instruction")
            
            if not question:
                continue # Skip if we can't find a question text
                
            if question in existing_questions:
                continue
                
            request_body = {
                "contents": [{"parts": [{"text": question}]}],
                "generationConfig": {
                    "thinking_config": {
                        "include_thoughts": True,
                        "thinking_level": "HIGH"
                    }
                }
            }
            
            requests.append({
                "custom_id": f"req_{count}",
                "request": request_body,
                "original_question": question
            })
            count += 1

        if not requests:
            logger.info("No new requests generated.")
            return None

        # Save mapping to state
        self.state["request_mapping"] = {r["custom_id"]: r["original_question"] for r in requests}
        self._save_state()

        # Write batch file
        with open(self.config.batch_input_gen_file, "w", encoding="utf-8") as f:
            for r in requests:
                line_obj = {"custom_id": r["custom_id"], "request": r["request"]}
                f.write(json.dumps(line_obj) + "\n")
                
        return self.config.batch_input_gen_file

    def submit_batch(self, file_path: Path, model_id: str) -> str:
        """Uploads a file and creating a batch job."""
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
            logger.info(f"Status: {batch_job.state}")
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

    def process_generation_results(self, results_text: str) -> tuple[List[Dict], List[Dict]]:
        """Parses generation results into processed samples and those needing summarization."""
        logger.info("Processing generation results...")
        mapping = self.state.get("request_mapping", {})
        
        lines = results_text.strip().split("\n")
        processed_samples = []
        to_summarize = []
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                
                if not custom_id or custom_id not in mapping:
                    logger.warning(f"Unknown custom_id in result: {custom_id}")
                    continue
                    
                question = mapping[custom_id]
                response = result.get("response", {})
                
                if "candidates" not in response:
                    if "error" in response:
                        logger.error(f"Error in generation for {custom_id}: {response['error']}")
                    continue

                candidate = response["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                
                raw_logic = ""
                final_answer = ""
                
                for part in parts:
                    if part.get("thought"):
                        raw_logic += part.get("text", "")
                    else:
                        final_answer += part.get("text", "")
                
                sample_data = {
                    "question": question,
                    "raw_logic": raw_logic,
                    "final_answer": final_answer
                }
                
                # Logic to determine if summarization is needed
                if len(raw_logic.split()) > 100:
                    to_summarize.append(sample_data)
                else:
                    if not raw_logic:
                        logger.warning(f"Skipping sample {custom_id} due to missing reasoning.")
                        continue

                    formatted_output = f"<draft> {raw_logic} </draft> #### {final_answer}"
                    processed_samples.append({
                        "instruction": question,
                        "input": "",
                        "output": formatted_output
                    })
                    
            except Exception as e:
                logger.error(f"Error processing line {i}: {e}")
                continue
                
        return processed_samples, to_summarize

    def prepare_summarization_batch(self, to_summarize: List[Dict]) -> Path:
        """Prepares a batch file for summarizing long reasoning chains."""
        logger.info(f"Preparing summarization batch for {len(to_summarize)} samples...")
        
        requests = []
        mapping = {}
        
        for i, sample in enumerate(to_summarize):
            req_id = f"sum_{i}"
            
            summary_prompt = f"""
            Summarize the following reasoning process into 3-5 concise steps (max 5 words each).
            Format the output as a single line with numbered steps separated by arrows, wrapped in <draft> tags.
            Example: <draft> 1. Step one -> 2. Step two -> 3. Step three </draft>
            
            Reasoning:
            {sample['raw_logic']}
            """
            
            request_body = {
                "contents": [{"parts": [{"text": summary_prompt}]}]
            }
            
            requests.append({
                "custom_id": req_id,
                "request": request_body
            })
            mapping[req_id] = sample
            
        self.state["summarize_mapping"] = mapping
        self._save_state()
        
        with open(self.config.batch_input_sum_file, "w", encoding="utf-8") as f:
            for r in requests:
                line_obj = {"custom_id": r["custom_id"], "request": r["request"]}
                f.write(json.dumps(line_obj) + "\n")
                
        return self.config.batch_input_sum_file

    def process_summarization_results(self, results_text: str) -> List[Dict]:
        """Parses summarization results and merges them with original samples."""
        logger.info("Processing summarization results...")
        mapping = self.state.get("summarize_mapping", {})
        
        lines = results_text.strip().split("\n")
        final_samples = []
        
        for i, line in enumerate(lines):
            try:
                result = json.loads(line)
                custom_id = result.get("custom_id")
                
                if not custom_id or custom_id not in mapping:
                    continue

                response = result.get("response", {})
                if "candidates" not in response:
                    if "error" in response:
                        logger.error(f"Error in summarization for {custom_id}: {response['error']}")
                    continue

                candidate = response["candidates"][0]
                summary = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])).strip()
                
                # Ensure <draft> tags format
                if not summary.startswith("<draft>"):
                    summary = f"<draft> {summary}"
                if not summary.endswith("</draft>"):
                    summary = f"{summary} </draft>"
                
                original_sample = mapping[custom_id]
                formatted_output = f"{summary} #### {original_sample['final_answer']}"
                
                final_samples.append({
                    "instruction": original_sample["question"],
                    "input": "",
                    "output": formatted_output
                })
                
            except Exception as e:
                logger.error(f"Error processing summary line {i}: {e}")
                continue
                
        return final_samples

    def run_step_1(self):
        """Step 1: Prepare and Submit Generation Batch"""
        batch_file = self.prepare_generation_batch()
        if batch_file:
            batch_name = self.submit_batch(batch_file, self.config.model_id)
            self.state["generation_batch_name"] = batch_name
            self.state["step"] = "1_done"
            self._save_state()
            logger.info(f"Generation batch submitted: {batch_name}. Run step 'status' to monitor.")
        else:
            logger.info("Nothing to submit.")

    def run_step_2(self):
        """Step 2: Process Generation and Submit Summarization"""
        if "generation_batch_name" not in self.state:
            logger.error("No generation batch found in state.")
            return

        batch_name = self.state["generation_batch_name"]
        job = self.check_batch_status(batch_name)
        
        if str(job.state) == "JobState.JOB_STATE_SUCCEEDED":
            results_text = self.download_results(job)
            if not results_text:
                return

            processed, to_summarize = self.process_generation_results(results_text)
            
            if processed:
                self._append_to_output(processed)
                logger.info(f"Saved {len(processed)} samples directly.")

            if to_summarize:
                batch_file = self.prepare_summarization_batch(to_summarize)
                sum_batch_name = self.submit_batch(batch_file, self.config.model_id)
                self.state["summarization_batch_name"] = sum_batch_name
                self.state["step"] = "2_done"
                self._save_state()
                logger.info(f"Summarization batch submitted: {sum_batch_name}")
            else:
                logger.info("No summarization needed. Done.")
                self.state["step"] = "complete"
                self._save_state()
        else:
            logger.info(f"Generation batch is not ready: {job.state}")

    def run_step_3(self):
        """Step 3: Process Summarization"""
        if "summarization_batch_name" not in self.state:
            logger.error("No summarization batch found.")
            return

        batch_name = self.state["summarization_batch_name"]
        job = self.check_batch_status(batch_name)
        
        if str(job.state) == "JobState.JOB_STATE_SUCCEEDED":
            results_text = self.download_results(job)
            if not results_text:
                return

            final_samples = self.process_summarization_results(results_text)
            if final_samples:
                self._append_to_output(final_samples)
                logger.info(f"Saved {len(final_samples)} summarized samples.")
                
            self.state["step"] = "complete"
            self._save_state()
            logger.info("All done.")
        else:
            logger.info(f"Summarization batch is not ready: {job.state}")

    def check_status(self):
        """Checks status of all active batches."""
        if "generation_batch_name" in self.state:
            job = self.check_batch_status(self.state["generation_batch_name"])
            logger.info(f"Generation Batch ({self.state['generation_batch_name']}): {job.state}")
        if "summarization_batch_name" in self.state:
            job = self.check_batch_status(self.state["summarization_batch_name"])
            logger.info(f"Summarization Batch ({self.state['summarization_batch_name']}): {job.state}")

    def _append_to_output(self, data: List[Dict]):
        """Helper to append data to the output file."""
        with open(self.config.output_file, "a", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    parser = argparse.ArgumentParser(description="Generate Chain of Draft data using Gemini Batch API")
    parser.add_argument("--step", type=str, choices=["1", "2", "3", "status"], required=True, help="Step to execute")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash-thinking-exp-1219", help="Model ID to use")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name to use (e.g., 'gsm8k', 'openai/gsm8k')")
    parser.add_argument("--limit", type=int, default=1000, help="Total samples to generate")
    args = parser.parse_args()

    config = GenerationConfig(
        api_key=api_key,
        model_id=args.model,
        dataset_name=args.dataset,
        target_total=args.limit
    )
    
    generator = ChainOfDraftGenerator(config)

    if args.step == "1":
        generator.run_step_1()
    elif args.step == "2":
        generator.run_step_2()
    elif args.step == "3":
        generator.run_step_3()
    elif args.step == "status":
        generator.check_status()

if __name__ == "__main__":
    main()
