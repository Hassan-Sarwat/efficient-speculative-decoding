import os
import json
import logging
import argparse
import time
import sys
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Add project root to sys.path to allow imports from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generation.prompts import SUMMARIZATION_PROMPT
from data_generation.batch_client import BatchClient

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

# ============================================================================
# NEW: ProcessingMetrics Class - Add this at the top after imports
# ============================================================================

@dataclass
class ProcessingMetrics:
    """Tracks statistics and errors during batch processing"""
    
    # Generation metrics
    total_lines_processed: int = 0
    successful_generated: int = 0
    successful_summarized: int = 0
    
    # Token usage (Split)
    cot_prompt_tokens: int = 0
    cot_candidate_tokens: int = 0
    cod_prompt_tokens: int = 0
    cod_candidate_tokens: int = 0
    
    # Error counters
    errors: Dict[str, int] = field(default_factory=lambda: {
        'json_decode': 0,
        'empty_reasoning': 0,
        'api_error': 0,
        'missing_id': 0,
        'missing_answer': 0,
        'format_error': 0
    })
    
    # Processing details
    total_sent_to_summarization: int = 0
    
    def log_error(self, error_type: str, details: str = ""):
        """Log an error occurrence"""
        if error_type in self.errors:
            self.errors[error_type] += 1
        else:
            self.errors['format_error'] += 1
        
        if details:
            logger.debug(f"{error_type}: {details}")
    
    def log_success(self, processing_type: str):
        """Log a successful processing"""
        if processing_type == "generation":
            self.successful_generated += 1
        elif processing_type == "summarization":
            self.successful_summarized += 1
            
    def update_usage(self, prompt_tokens: int, candidate_tokens: int, stage: str = "cot"):
        """Update token usage counts. stage: 'cot' or 'cod'"""
        if stage == "cot":
            self.cot_prompt_tokens += prompt_tokens
            self.cot_candidate_tokens += candidate_tokens
        else:
            self.cod_prompt_tokens += prompt_tokens
            self.cod_candidate_tokens += candidate_tokens
    
    def get_cost_estimate(self, model_id: str = "gemini-3-pro-preview") -> float:
        """
        Estimate cost based on token usage. 
        Using Gemini 1.5 Pro rates as a baseline for 'Pro' models:
        Input: $1.25 / 1M tokens
        Output: $5.00 / 1M tokens
        """
        # Pricing per 1M tokens
        PRICING = {
            "gemini-3-pro-preview": {"input": 1.00, "output": 6.00}, # Assumed Batch Pro-tier pricing
        }
        
        # Match model ID loosely
        pricing_tier = PRICING[model_id]
        for key in PRICING:
            if key in model_id:
                pricing_tier = PRICING[key]
                break
                
        cot_input_cost = (self.cot_prompt_tokens / 1_000_000) * pricing_tier["input"]
        cot_output_cost = (self.cot_candidate_tokens / 1_000_000) * pricing_tier["output"]
        
        cod_input_cost = (self.cod_prompt_tokens / 1_000_000) * pricing_tier["input"]
        cod_output_cost = (self.cod_candidate_tokens / 1_000_000) * pricing_tier["output"]
        
        return cot_input_cost + cot_output_cost, cod_input_cost + cod_output_cost

    def get_summary(self) -> str:
        """Generate a human-readable summary"""
        total_errors = sum(self.errors.values())
        success_rate = 0
        if self.total_lines_processed > 0:
            success_rate = ((self.successful_generated + self.successful_summarized) / 
                           self.total_lines_processed * 100)
        
        cot_cost, cod_cost = self.get_cost_estimate("gemini-3-pro-preview")
        total_cost = cot_cost + cod_cost
        
        summary = f"""
{'='*60}
PROCESSING METRICS SUMMARY
{'='*60}

Overall Stats:
  Total Lines Processed: {self.total_lines_processed}
  Successfully Generated (CoT): {self.successful_generated}
  Sent to Summarization: {self.total_sent_to_summarization}
  Successfully Summarized (CoD): {self.successful_summarized}
  Success Rate: {success_rate:.1f}%

Cost & Usage Analysis (Est.):
  ----------------------------------------
  Chain of Thought (Generation):
    Input Tokens: {self.cot_prompt_tokens:,}
    Output Tokens: {self.cot_candidate_tokens:,}
    Est. Cost: ${cot_cost:.4f}
  
  ----------------------------------------
  Chain of Draft (Summarization):
    Input Tokens: {self.cod_prompt_tokens:,}
    Output Tokens: {self.cod_candidate_tokens:,}
    Est. Cost: ${cod_cost:.4f}

  ----------------------------------------
  TOTAL COST: ${total_cost:.4f}
  (Based on typical Pro-tier pricing: $1.00/1M in, $6.00/1M out)

Error Breakdown (Total: {total_errors}):
"""
        for error_type, count in self.errors.items():
            if count > 0:
                summary += f"  - {error_type}: {count}\n"
        
        summary += f"\n{'='*60}\n"
        return summary
    
    def save_to_file(self, filepath: Path):
        """Save metrics to JSON file for later analysis"""
        metrics_dict = {
            "total_lines_processed": self.total_lines_processed,
            "successful_generated": self.successful_generated,
            "successful_summarized": self.successful_summarized,
            "total_sent_to_summarization": self.total_sent_to_summarization,
            "total_sent_to_summarization": self.total_sent_to_summarization,
            "cot_prompt_tokens": self.cot_prompt_tokens,
            "cot_candidate_tokens": self.cot_candidate_tokens,
            "cod_prompt_tokens": self.cod_prompt_tokens,
            "cod_candidate_tokens": self.cod_candidate_tokens,
            "estimated_cost": self.get_cost_estimate("gemini-3-pro-preview"), # Tuple
            "errors": self.errors,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ProcessingMetrics':
        """Load metrics from existing JSON file"""
        if not filepath.exists():
            return cls()
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            metrics = cls()
            metrics.total_lines_processed = data.get("total_lines_processed", 0)
            metrics.successful_generated = data.get("successful_generated", 0)
            metrics.successful_summarized = data.get("successful_summarized", 0)
            metrics.total_sent_to_summarization = data.get("total_sent_to_summarization", 0)
            metrics.cot_prompt_tokens = data.get("cot_prompt_tokens", 0)
            metrics.cot_candidate_tokens = data.get("cot_candidate_tokens", 0)
            metrics.cod_prompt_tokens = data.get("cod_prompt_tokens", 0)
            metrics.cod_candidate_tokens = data.get("cod_candidate_tokens", 0)
            metrics.errors = data.get("errors", metrics.errors)
            
            return metrics
        except Exception as e:
            logger.warning(f"Could not load existing metrics from {filepath}: {e}")
            return cls()

def get_existing_instructions(filepath: Path) -> set:
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

# ============================================================================
# MODIFIED: process_generation_results - Now tracks metrics
# ============================================================================

def process_generation_results(
    results_text: str, 
    mapping: Dict[str, str],
    metrics: ProcessingMetrics  # NEW PARAMETER
) -> tuple[List[Dict], List[Dict]]:
    """Parses generation results into processed samples and those needing summarization."""
    lines = results_text.strip().split("\n")
    cod_samples = [] # Chain of Draft (concise)
    cot_samples = [] # Chain of Thought (full reasoning)
    to_summarize = []
    
    for i, line in enumerate(lines):
        metrics.total_lines_processed += 1  # Track every line
        
        if not line.strip():
            continue
        
        try:
            result = json.loads(line)
            
            # --- Capture Token Usage ---
            # --- Capture Token Usage ---
            # Try top-level first, then response-level
            raw_response = result.get("response", {})
            if "body" in raw_response:
                response = raw_response["body"]
            else:
                response = raw_response

            # Usage is inside the normalized response
            usage = response.get("usageMetadata", {})
            p_tokens = usage.get("promptTokenCount", 0)
            c_tokens = usage.get("candidatesTokenCount", 0)
            metrics.update_usage(p_tokens, c_tokens, stage="cot")
            # ---------------------------

            custom_id = result.get("key")
            
            if not custom_id or custom_id not in mapping:
                metrics.log_error('missing_id', f"Line {i}: custom_id={custom_id}")
                continue
                
            question = mapping[custom_id]
            response = result.get("response", {})
            
            if "candidates" not in response:
                if "error" in response:
                    metrics.log_error('api_error', f"{custom_id}: {response['error']}")
                    logger.error(f"Error in generation for {custom_id}: {response['error']}")
                else:
                    metrics.log_error('api_error', f"{custom_id}: No candidates in response")
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

            # Cleanup final answer to prevent double ####
            final_answer = final_answer.replace("####", "").strip()
            
            # Validation: Check for empty reasoning
            if not raw_logic:
                metrics.log_error('empty_reasoning', f"{custom_id}")
                logger.warning(f"Skipping sample {custom_id} due to missing reasoning.")
                continue
            
            # Validation: Check for missing answer
            if not final_answer:
                metrics.log_error('missing_answer', f"{custom_id}")
                logger.warning(f"Skipping sample {custom_id} due to missing final answer.")
                continue
            
            sample_data = {
                "question": question,
                "raw_logic": raw_logic,
                "final_answer": final_answer,
                "custom_id": custom_id 
            }
            
            metrics.log_success("generation")
            
            # Logic to determine if summarization is needed
            # MODIFIED: Always summarize to ensure consistent CoD style
            to_summarize.append(sample_data)
            metrics.total_sent_to_summarization += 1

            # Always add to refined CoT samples (raw_logic is the CoT)
            cot_samples.append({
                "instruction": question,
                "input": "",
                "output": f"<thought> {raw_logic} </thought> #### {final_answer}"
            })
                
        except json.JSONDecodeError as e:
            metrics.log_error('json_decode', f"Line {i}: {str(e)}")
            logger.error(f"JSON decode error on line {i}: {e}")
            continue
        except Exception as e:
            metrics.log_error('format_error', f"Line {i}: {str(e)}")
            logger.error(f"Unexpected error processing line {i}: {e}")
            continue
            
    return cod_samples, cot_samples, to_summarize

# ============================================================================
# MODIFIED: process_summarization_results - Now tracks metrics
# ============================================================================

def process_summarization_results(
    results_text: str, 
    mapping: Dict[str, Dict],
    metrics: ProcessingMetrics  # NEW PARAMETER
) -> List[Dict]:
    """Parses summarization results and merges them with original samples."""
    lines = results_text.strip().split("\n")
    final_samples = []
    
    for i, line in enumerate(lines):
        metrics.total_lines_processed += 1
        
        try:
            result = json.loads(line)
            
            # --- Capture Token Usage ---
            # --- Capture Token Usage ---
            # Try top-level first, then response-level
            raw_response = result.get("response", {})
            if "body" in raw_response:
                response = raw_response["body"]
            else:
                response = raw_response

            usage = response.get("usageMetadata", {})
            p_tokens = usage.get("promptTokenCount", 0)
            c_tokens = usage.get("candidatesTokenCount", 0)
            metrics.update_usage(p_tokens, c_tokens, stage="cod")
            # ---------------------------
            
            custom_id = result.get("key")
            
            if not custom_id or custom_id not in mapping:
                metrics.log_error('missing_id', f"Summarization line {i}")
                continue

            response = result.get("response", {})
            if "candidates" not in response:
                metrics.log_error('api_error', f"Summarization {custom_id}: No candidates")
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
            
            metrics.log_success("summarization")
            
        except json.JSONDecodeError as e:
            metrics.log_error('json_decode', f"Summarization line {i}: {str(e)}")
            continue
        except Exception as e:
            metrics.log_error('format_error', f"Summarization line {i}: {str(e)}")
            logger.error(f"Unexpected error processing line {i}: {e}")
            continue
            
    return final_samples

# ============================================================================
# MODIFIED: main - Now creates and uses metrics
# ============================================================================

def main():
    load_dotenv()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
        
    parser = argparse.ArgumentParser(description="Process Chain of Draft Batch Results")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", 
                       help="Dataset name to identify state file")
    parser.add_argument("--temp_dir", type=str, default="tmp", 
                       help="Directory for state files")
    parser.add_argument("--output_dir", type=str, default="data", 
                       help="Directory for final output")
    parser.add_argument("--file_suffix", type=str, help="Custom suffix to identify state file")
    parser.add_argument("--overwrite_metrics", action="store_true", help="Overwrite existing metrics file instead of appending")
    args = parser.parse_args()

    safe_name = args.dataset.replace("/", "_")
    
    if args.file_suffix:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}_{args.file_suffix}.json"
        metrics_file = Path(args.output_dir) / f"metrics_{safe_name}_{args.file_suffix}.json"
    else:
        batch_state_file = Path(args.temp_dir) / f"batch_state_{safe_name}.json"
        metrics_file = Path(args.output_dir) / f"metrics_{safe_name}.json"
    
    if not batch_state_file.exists():
        logger.error(f"State file {batch_state_file} not found.")
        return

    with open(batch_state_file, "r") as f:
        state = json.load(f)

    if "batches" not in state:
        logger.info("No batches found in state.")
        return

    # ========================================================================
    # NEW: Create ProcessingMetrics instance
    # ========================================================================
    if metrics_file.exists() and not args.overwrite_metrics:
        logger.info(f"Loading existing metrics from {metrics_file}")
        metrics = ProcessingMetrics.load_from_file(metrics_file)
    else:
        if args.overwrite_metrics and metrics_file.exists():
            logger.info(f"Overwriting existing metrics file: {metrics_file}")
        metrics = ProcessingMetrics()
    
    client = BatchClient(api_key=api_key)
    updated_batches = []
    
    for batch in state["batches"]:
        if batch["status"] == "done":
            updated_batches.append(batch)
            continue
            
        batch_name = batch["batch_name"]
        job = client.check_batch_status(batch_name)
        
        logger.info(f"Batch {batch_name} ({batch['type']}): {job.state}")
        
        if str(job.state) == "JobState.JOB_STATE_SUCCEEDED":
            results_text = client.download_results(job)
            if not results_text:
                updated_batches.append(batch)
                continue
                
            if batch["type"] == "generation":
                target_output_file = Path(batch.get("output_file", Path(args.output_dir) / f"cob_data_{safe_name}.jsonl"))
                
                # Derive CoT and CoD filenames from the target output file (which is the CoD file usually)
                # target_output_file typically looks like "data/cob_xy.jsonl"
                # We want "data/cot_xy.jsonl" and "data/cod_xy.jsonl"
                
                # If file starts with cot_, we keep it for CoT and create cob_ for CoD
                filename = target_output_file.name
                if filename.startswith("cot_"):
                   cot_filename = filename
                   cod_filename = filename.replace("cot_", "cob_", 1)
                elif filename.startswith("cob_"): # Legacy support or user override
                   cod_filename = filename 
                   cot_filename = filename.replace("cob_", "cot_", 1)
                else:
                   # Fallback
                   cod_filename = f"cob_{filename}"
                   cot_filename = f"cot_{filename}"
                   
                cod_file = target_output_file.parent / cod_filename
                cot_file = target_output_file.parent / cot_filename

                # MODIFIED: Pass metrics to processing function
                cod_ready, cot_ready, to_summarize = process_generation_results(
                    results_text, 
                    batch["mapping"],
                    metrics  # NEW
                )
                
                # Save CoT samples (Always save full reasoning)
                if cot_ready:
                    existing_cot = get_existing_instructions(cot_file)
                    new_cot = [item for item in cot_ready if item["instruction"] not in existing_cot]
                    
                    if new_cot:
                        with open(cot_file, "a", encoding="utf-8") as f:
                            for item in new_cot:
                                f.write(json.dumps(item) + "\n")
                        logger.info(f"Saved {len(new_cot)} CoT samples to {cot_file}. (Skipped {len(cot_ready) - len(new_cot)} duplicates)")
                    else:
                        logger.info(f"All {len(cot_ready)} CoT samples were duplicates. Skipped writing.")

                # Save ready CoD samples (short ones)
                if cod_ready:
                    existing_cod = get_existing_instructions(cod_file)
                    new_cod = [item for item in cod_ready if item["instruction"] not in existing_cod]

                    if new_cod:
                        with open(cod_file, "a", encoding="utf-8") as f:
                            for item in new_cod:
                                f.write(json.dumps(item) + "\n")
                        logger.info(f"Saved {len(new_cod)} CoD samples to {cod_file}. (Skipped {len(cod_ready) - len(new_cod)} duplicates)")

                # Submit summarization if needed
                if to_summarize:
                    logger.info(f"Submitting summarization for {len(to_summarize)} samples.")
                    
                    sum_requests = []
                    sum_mapping = {}
                    
                    for i, sample in enumerate(to_summarize):
                        req_id = f"sum_{batch_name}_{i}"
                        summary_prompt = SUMMARIZATION_PROMPT.format(question=sample['question'], raw_logic=sample['raw_logic'])
                        
                        request_body = {
                            "contents": [{"parts": [{"text": summary_prompt}]}]
                        }
                        
                        sum_requests.append({
                            "key": req_id,
                            "request": request_body
                        })
                        sum_mapping[req_id] = sample
                    
                    safe_batch_name = batch_name.replace("/", "_")
                    sum_file = Path(args.temp_dir) / f"batch_input_sum_{safe_batch_name}.jsonl"
                    with open(sum_file, "w", encoding="utf-8") as f:
                        for r in sum_requests:
                            line_obj = {"key": r["key"], "request": r["request"]}
                            f.write(json.dumps(line_obj) + "\n")
                    
                    try:
                        sum_batch_name = client.submit_batch(sum_file, batch["model"])
                        updated_batches.append({
                            "batch_name": sum_batch_name,
                            "type": "summarization",
                            "mapping": sum_mapping,
                            "status": "submitted",
                            "timestamp": time.time(),
                            "model": batch["model"],
                            "output_file": str(target_output_file) # Propagate intended output info
                        })
                        batch["status"] = "done"
                        updated_batches.append(batch)
                        
                    except Exception as e:
                        logger.error(f"Failed to submit summary batch: {e}")
                        updated_batches.append(batch)
                else:
                    batch["status"] = "done"
                    updated_batches.append(batch)
            
            elif batch["type"] == "summarization":
                target_output_file = Path(batch.get("output_file", Path(args.output_dir) / f"cob_data_{safe_name}.jsonl"))
                filename = target_output_file.name
                if filename.startswith("cob_"):
                   cod_filename = filename
                elif filename.startswith("cot_"):
                   cod_filename = filename.replace("cot_", "cob_", 1)
                else:
                   cod_filename = f"cob_{filename}"
                cod_file = target_output_file.parent / cod_filename

                # MODIFIED: Pass metrics to processing function
                final_samples = process_summarization_results(
                    results_text, 
                    batch["mapping"],
                    metrics  # NEW
                )
                
                if final_samples:
                    existing_cod = get_existing_instructions(cod_file)
                    new_samples = [item for item in final_samples if item["instruction"] not in existing_cod]
                    
                    if new_samples:
                        with open(cod_file, "a", encoding="utf-8") as f:
                            for item in new_samples:
                                f.write(json.dumps(item) + "\n")
                        logger.info(f"Saved {len(new_samples)} summarized samples to {cod_file}. (Skipped {len(final_samples) - len(new_samples)} duplicates)")
                    else:
                        logger.info(f"All {len(final_samples)} summarized samples were duplicates. Skipped writing.")
                
                batch["status"] = "done"
                updated_batches.append(batch)
        
        else:
            updated_batches.append(batch)

    state["batches"] = updated_batches
    with open(batch_state_file, "w") as f:
        json.dump(state, f, indent=2)
    
    # ========================================================================
    # NEW: Print and save metrics summary
    # ========================================================================
    print(metrics.get_summary())
    metrics.save_to_file(metrics_file)

if __name__ == "__main__":
    main()