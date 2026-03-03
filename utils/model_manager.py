# src/modules/llm/model_manager.py
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from loguru import logger
from typing import Dict, Tuple, Optional, Any, List, Union

class HeterogeneousLogitsProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor to apply different sampling parameters (Temperature, Repetition Penalty, Top-P)
    to each sample in a batch.
    """
    def __init__(self, 
                 temperatures: Optional[torch.Tensor] = None, 
                 repetition_penalties: Optional[torch.Tensor] = None,
                 top_ps: Optional[torch.Tensor] = None,
                 pad_token_id: Optional[int] = None,
                 device: str = "cpu"):
        self.temperatures = temperatures
        self.repetition_penalties = repetition_penalties
        self.top_ps = top_ps
        self.pad_token_id = pad_token_id
        self.device = device

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        # scores: (batch_size, vocab_size) - logits before softmax

        # Ensure tensors are on the correct device
        device = scores.device

        # 1. Apply Repetition Penalty (Heterogeneous)
        if self.repetition_penalties is not None:
            if self.repetition_penalties.device != device:
                self.repetition_penalties = self.repetition_penalties.to(device)
            
            # Gather logits for the tokens in input_ids
            score = torch.gather(scores, 1, input_ids)
            
            # Expand penalties to match input_ids shape: (B, 1) -> (B, Seq_Len)
            penalties_expanded = self.repetition_penalties.unsqueeze(1).expand_as(score).to(scores.dtype)
            
            # Calculate penalized scores
            # if score < 0: score * penalty
            # if score > 0: score / penalty
            penalized_score = torch.where(score < 0, score * penalties_expanded, score / penalties_expanded)
            
            # CRITICAL FIX: Mask out pad_token_id from penalty.
            # If pad_token_id == eos_token_id (common in Llama/Qwen), penalizing PAD means 
            # penalizing EOS, which prevents the model from stopping if Prompt has padding.
            if self.pad_token_id is not None:
                mask = (input_ids == self.pad_token_id) # True where token IS pad
                # Restore original score where input is pad; use penalized otherwise
                score = torch.where(mask, score, penalized_score)
            else:
                score = penalized_score
            
            # Scatter back
            scores.scatter_(1, input_ids, score)

        # 2. Apply Temperature (Heterogeneous)
        if self.temperatures is not None:
            if self.temperatures.device != device:
                self.temperatures = self.temperatures.to(device)
            
            # scores: (B, V), temps: (B,) -> (B, 1)
            temps_expanded = self.temperatures.unsqueeze(1).to(scores.dtype)
            scores.div_(temps_expanded)

        # 3. Apply Top-P (Heterogeneous)
        if self.top_ps is not None:
            if self.top_ps.device != device:
                self.top_ps = self.top_ps.to(device) # (B,)

            # Sort logits to calculate cumulative probs
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (top_p)
            sorted_indices_to_remove = cumulative_probs > self.top_ps.unsqueeze(1)
            
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            
            # Set removed logits to -inf
            scores = scores.masked_fill(indices_to_remove, float('-inf'))

        return scores

class LocalLLMManager:
    """
    Singleton manager for Local LLMs to ensure models are loaded only once
    and kept in VRAM.
    """
    _instance = None
    _models: Dict[str, Any] = {}
    _tokenizers: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalLLMManager, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path: str, device: str = "cuda:0", dtype: str = "auto") -> None:
        """
        Loads a model into memory if not already loaded.

        Args:
            model_path: Local path to the model directory.
            device: Target device (e.g., "cuda:0", "cuda:1").
            dtype: Loading precision ("float16", "bfloat16", or "auto").
        """
        if model_path in self._models:
            # Already loaded
            return

        logger.info(f"[LocalLLM] Loading model from {model_path} to {device} ({dtype})...")

        try:
            torch_dtype = torch.float16
            if dtype == "bfloat16":
                torch_dtype = torch.bfloat16
            elif dtype == "auto":
                torch_dtype = "auto"

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
            
            # Ensure padding token exists for batch generation
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # Left padding is required for generation
            tokenizer.padding_side = "left"

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device, # Use accelerate's device_map for explicit placement
                local_files_only=True,
                trust_remote_code=True
            )

            self._models[model_path] = model
            self._tokenizers[model_path] = tokenizer
            
            # Log detailed model info
            num_params = sum(p.numel() for p in model.parameters()) / 1e9
            mem_used = "N/A"
            if torch.cuda.is_available() and "cuda" in str(model.device):
                mem_used = f"{torch.cuda.memory_allocated(model.device) / 1024**3:.2f} GB"
            max_len = getattr(model.config, "max_position_embeddings", "unknown")

            logger.info(
                f"[LocalLLM] Successfully loaded {model_path}\n"
                f"    - Device: {model.device}\n"
                f"    - Dtype: {model.dtype}\n"
                f"    - Params: {num_params:.2f}B\n"
                f"    - VRAM: {mem_used}\n"
                f"    - Context: {max_len}"
            )

        except Exception as e:
            logger.error(f"[LocalLLM] Failed to load model {model_path}: {e}")
            raise e

    def generate(self, 
                 model_path: str, 
                 prompt: Any,  # str or List[Dict]
                 max_new_tokens: int = 256, 
                 temperature: float = 0.7, 
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1) -> str:
        """
        Generates text using the specified loaded model (Single sample).
        """
        if model_path not in self._models:
            raise RuntimeError(f"Model {model_path} not loaded! Call load_model first.")

        model = self._models[model_path]
        tokenizer = self._tokenizers[model_path]
        device = model.device

        # Handle Chat Template
        text_input = self._apply_chat_template(tokenizer, prompt)
        
        inputs = tokenizer(text_input, return_tensors="pt").to(device)

        # Dynamic parameters
        do_sample = temperature > 0.0

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the new tokens
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        del inputs, outputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return text

    def generate_batch(self,
                       model_path: str,
                       prompts: List[Any],  # List[str] or List[List[Dict]]
                       temperatures: List[float],
                       repetition_penalties: List[float],
                       max_new_tokens: int = 256,
                       top_ps: Optional[List[float]] = None) -> List[str]:
        """
        Generates text for a batch of prompts, supporting per-sample sampling parameters.
        Supports fully heterogeneous sampling for Temperature, Repetition Penalty, and Top-P.
        """
        if model_path not in self._models:
            raise RuntimeError(f"Model {model_path} not loaded! Call load_model first.")

        model = self._models[model_path]
        tokenizer = self._tokenizers[model_path]
        device = model.device
        
        # Enforce left padding for generation
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token

        # 1. Prepare batch inputs
        text_inputs = [self._apply_chat_template(tokenizer, p) for p in prompts]
        
        # Tokenize with padding
        inputs = tokenizer(text_inputs, return_tensors="pt", padding=True).to(device)
        
        # 2. Prepare Heterogeneous Logits Processor
        # Convert params to tensors on device
        temp_tensor = torch.tensor(temperatures, device=device, dtype=torch.float32)
        rep_tensor = torch.tensor(repetition_penalties, device=device, dtype=torch.float32)
        
        # Handle heterogeneous top-p if provided
        top_p_tensor = None
        if top_ps and len(top_ps) > 0:
             # Normalize length if needed (though caller should handle)
             if len(top_ps) < len(prompts):
                 top_ps = top_ps + [0.9] * (len(prompts) - len(top_ps))
             top_p_tensor = torch.tensor(top_ps, device=device, dtype=torch.float32)

        # Important: Handle cases where temperature is very close to 0 (greedy)
        # The processor will divide by temp, so clamp min temp to small epsilon to avoid inf
        temp_tensor = torch.clamp(temp_tensor, min=1e-4)
        
        processor = HeterogeneousLogitsProcessor(
            temperatures=temp_tensor,
            repetition_penalties=rep_tensor,
            top_ps=top_p_tensor,
            pad_token_id=tokenizer.pad_token_id,
            device=device
        )
        logits_processor = LogitsProcessorList([processor])

        # 3. Generate
        # We set global temperature=1.0 and repetition_penalty=1.0 because our custom processor handles them.
        # We also set global top_p=1.0 (disabled) because our processor now handles heterogeneous top_p.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,         # Always sample
                temperature=1.0,        # Disabled global scaling
                repetition_penalty=1.0, # Disabled global penalty
                top_p=1.0,              # Disabled global top_p, handled by processor
                logits_processor=logits_processor,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Decode batch
        results = []
        input_len = inputs.input_ids.shape[1]
        for i in range(len(prompts)):
            # Slice out only new tokens
            gen_ids = outputs[i][input_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(text)
            
        del inputs, outputs, processor, logits_processor, temp_tensor, rep_tensor
        if top_p_tensor is not None:
            del top_p_tensor
        gc.collect()
        torch.cuda.empty_cache()

        return results

    def _apply_chat_template(self, tokenizer, prompt):
        if isinstance(prompt, list):
            try:
                return tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
        return prompt

# Global instance
model_manager = LocalLLMManager()
