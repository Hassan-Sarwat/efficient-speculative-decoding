import json
import logging
import time
from typing import Dict, List, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

class FormatConstants:
    """Formatting constants for output generation."""
    THOUGHT_TAG_OPEN = "<thought>"
    THOUGHT_TAG_CLOSE = "</thought>"
    DRAFT_TAG_OPEN = "<draft>"
    DRAFT_TAG_CLOSE = "</draft>"
    ANSWER_MARKER = "####"
    
    @classmethod
    def wrap_thought(cls, content: str) -> str:
        """Wrap content in thought tags."""
        return f"{cls.THOUGHT_TAG_OPEN} {content} {cls.THOUGHT_TAG_CLOSE}"
    
    @classmethod
    def wrap_draft(cls, content: str) -> str:
        """Wrap content in draft tags."""
        # Ensure tags are present
        if not content.strip().startswith(cls.DRAFT_TAG_OPEN):
            content = f"{cls.DRAFT_TAG_OPEN} {content}"
        if not content.strip().endswith(cls.DRAFT_TAG_CLOSE):
            content = f"{content} {cls.DRAFT_TAG_CLOSE}"
        return content
    
    @classmethod
    def format_with_answer(cls, reasoning: str, answer: str) -> str:
        """Format reasoning with answer marker."""
        return f"{reasoning} {cls.ANSWER_MARKER} {answer}"

# ============================================================================
# ProcessingMetrics Class
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
    
    def get_cost_estimate(self, model_id: str = "gemini-3-pro-preview") -> Tuple[float, float]:
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
        pricing_tier = PRICING.get("gemini-3-pro-preview") # Default
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

@dataclass
class TrainingSample:
    """Standard training sample format for fine-tuning."""
    instruction: str
    input: str
    output: str

@dataclass
class IntermediateSample:
    """Intermediate sample awaiting summarization."""
    question: str
    raw_logic: str
    final_answer: str
    custom_id: str

# ============================================================================
# Core Logic
# ============================================================================

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

def extract_token_usage(result: Dict[str, Any]) -> Tuple[int, int]:
    """
    Extract token usage from API result.
    
    Handles both top-level and nested response formats.
    
    Args:
        result: Raw API result dict
        
    Returns:
        Tuple of (prompt_tokens, candidate_tokens)
    """
    raw_response = result.get("response", {})
    
    # Handle nested body structure
    if "body" in raw_response:
        response = raw_response["body"]
    else:
        response = raw_response
    
    usage = response.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount", 0)
    candidate_tokens = usage.get("candidatesTokenCount", 0)
    
    return prompt_tokens, candidate_tokens

def process_generation_results(
    results_path: Path,
    mapping: Dict[str, str],
    metrics: ProcessingMetrics
) -> Tuple[List[TrainingSample], List[TrainingSample], List[IntermediateSample]]:
    """
    Parse generation results into structured samples.
    
    Returns:
        Tuple of:
        - cod_samples: Ready-to-use CoD samples (empty until summarization)
        - cot_samples: Chain of Thought samples with full reasoning
        - to_summarize: Samples needing summarization for CoD format
    """

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if not mapping:
        raise ValueError("Mapping dictionary is empty")
    
    if results_path.stat().st_size == 0:
        logger.warning(f"Results file is empty: {results_path}")
        return [], [], []
    
    cod_samples = [] # Chain of Draft (concise)
    cot_samples = [] # Chain of Thought (full reasoning)
    to_summarize = []

    with open(results_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            metrics.total_lines_processed += 1
            if not line.strip():
                continue
        
            try:
                result = json.loads(line)
                
                # --- Capture Token Usage ---
                # Try top-level first, then response-level
                p_tokens, c_tokens = extract_token_usage(result)
                metrics.update_usage(p_tokens, c_tokens, stage="cot")
                # ---------------------------

                # Still need response for candidate extraction
                raw_response = result.get("response", {})
                response = raw_response.get("body", raw_response)

                custom_id = result.get("key")
                
                if not custom_id or custom_id not in mapping:
                    metrics.log_error('missing_id', f"Line {i}: custom_id={custom_id}")
                    continue
                    
                question = mapping[custom_id]
                
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
                final_answer = final_answer.replace(FormatConstants.ANSWER_MARKER, "").strip()

                
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
                
                intermediate = IntermediateSample(
                    question=question,
                    raw_logic=raw_logic,
                    final_answer=final_answer,
                    custom_id=custom_id
                )
                to_summarize.append(intermediate)
                
                metrics.log_success("generation")
                
                # Logic to determine if summarization is needed
                # Always summarize to ensure consistent CoD style
                metrics.total_sent_to_summarization += 1

                thought_section = FormatConstants.wrap_thought(raw_logic)
                output = FormatConstants.format_with_answer(thought_section, final_answer)

                # Always add to refined CoT samples (raw_logic is the CoT)
                cot_sample = TrainingSample(
                    instruction=question,
                    input="",
                    output=output
                )
                cot_samples.append(cot_sample)
                    
            except json.JSONDecodeError as e:
                metrics.log_error('json_decode', f"Line {i}: {str(e)}")
                logger.error(f"JSON decode error on line {i}: {e}")
                continue
            except Exception as e:
                metrics.log_error('format_error', f"Line {i}: {str(e)}")
                logger.error(f"Unexpected error processing line {i}: {e}")
                continue
            
    return cod_samples, cot_samples, to_summarize

def process_summarization_results(
    results_path: Path, 
    mapping: Dict[str, IntermediateSample],  # ✅ Clear!
    metrics: ProcessingMetrics
) -> List[TrainingSample]:
    """Parses summarization results and merges them with original samples."""

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if not mapping:
        raise ValueError("Mapping dictionary is empty")
    
    if results_path.stat().st_size == 0:
        logger.warning(f"Results file is empty: {results_path}")
        return []
    
    final_samples = []

    with open(results_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            metrics.total_lines_processed += 1
            if not line.strip():
                continue
        
            try:
                result = json.loads(line)
                
                # --- Capture Token Usage ---
                # Try top-level first, then response-level
                p_tokens, c_tokens = extract_token_usage(result)
                metrics.update_usage(p_tokens, c_tokens, stage="cod")

                # Still need response for candidate extraction
                raw_response = result.get("response", {})
                response = raw_response.get("body", raw_response)

                # ---------------------------
                
                custom_id = result.get("key")
                
                if not custom_id or custom_id not in mapping:
                    metrics.log_error('missing_id', f"Summarization line {i}")
                    continue

                if "candidates" not in response:
                    metrics.log_error('api_error', f"Summarization {custom_id}: No candidates")
                    continue

                candidate = response["candidates"][0]
                summary = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])).strip()
                
                # Ensure <draft> tags format
                summary = FormatConstants.wrap_draft(summary)
                original = mapping[custom_id]
                formatted_output = FormatConstants.format_with_answer(summary, original.final_answer)
                
                sample = TrainingSample(
                    instruction=original.question,
                    input="",
                    output=formatted_output
                )
                final_samples.append(sample)
                
                metrics.log_success("summarization")
                
            except json.JSONDecodeError as e:
                metrics.log_error('json_decode', f"Summarization line {i}: {str(e)}")
                continue
            except Exception as e:
                metrics.log_error('format_error', f"Summarization line {i}: {str(e)}")
                logger.error(f"Unexpected error processing line {i}: {e}")
                continue
            
    return final_samples
