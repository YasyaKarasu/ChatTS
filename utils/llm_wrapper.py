import os
import json
import time
import requests
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from loguru import logger

# Conditional import for Local LLM Manager
try:
    try:
        from .model_manager import model_manager
    except (ImportError, ValueError):
        from model_manager import model_manager
except ImportError:
    model_manager = None

@dataclass
class LLMConfig:
    api_key: str
    model_name: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: int = 60
    max_retries: int = 3
    debug: bool = False
    default_max_tokens: int = 2048
    default_temperature: float = 0.2
    default_top_p: float = 0.9
    default_repetition_penalty: float = 1.05

    # New fields for Local LLM
    backend: str = "vllm"  # "api", "local", or "vllm"
    device_map: str = "cuda:0"
    dtype: str = "float16"

class GenerationOutput(str):
    """
    String subclass that carries token usage metadata.
    Behaves exactly like a string for legacy code.
    """
    def __new__(cls, text, usage: Dict[str, int] = None):
        obj = str.__new__(cls, text)
        obj.usage = usage or {"prompt_tokens": 0, "completion_tokens": 0}
        return obj

class ImprovedLLMWrapper:
    """
    LLM chat-completions wrapper supporting API (OpenRouter), Local (HuggingFace), and vLLM backends.

    Backend selection is controlled by `backend` argument ("api", "local", "vllm").
    """
    def __init__(self,
                 model_name: str,
                 api_key: Optional[str] = None,
                 belief_dim: Optional[int] = None,
                 base_url: str = "https://openrouter.ai/api/v1",
                 timeout: Optional[int] = None,
                 max_retries: int = 3,
                 debug: bool = False,
                 backend: str = "vllm",
                 device_map: str = "cuda:0",
                 dtype: str = "auto",
                 **kwargs):
        """
        Args:
            backend: "api" (OpenRouter), "vllm" (Local API), or "local" (HuggingFace).
            model_name: Model ID (API) or local path (Local).
            device_map: Target GPU for local model (e.g., "cuda:0").
        """
        # alias mapping
        if timeout is None and "timeout_s" in kwargs and kwargs["timeout_s"] is not None:
            try:
                timeout = int(kwargs["timeout_s"])
            except Exception:
                timeout = None

        if timeout is None:
            timeout = 60  # default

        if debug and kwargs:
            safe_keys = ", ".join(sorted(k for k in kwargs.keys()))
            logger.debug(f"[ImprovedLLMWrapper] Ignoring extra kwargs: {safe_keys}")

        # Handle vLLM defaults & Auto-discovery
        if backend == "vllm":
            # Try to load vllm_model_config.yaml to auto-configure port
            # Search paths: specific arg, current dir, config dir, or original script dir
            search_paths = [
                kwargs.get("vllm_config_path"),
                "vllm_model_config.yaml",
                "config/vllm_model_config.yaml",
                "scripts/vllm/vllm_model_config.yaml"
            ]
            
            config_path = None
            for p in search_paths:
                if p and os.path.exists(p):
                    config_path = p
                    break

            if not config_path:
                raise FileNotFoundError("[LLM Wrapper] 'vllm_model_config.yaml' not found in search paths. Strict configuration required for vLLM backend.")

            try:
                import yaml
                with open(config_path, 'r') as f:
                    model_config = yaml.safe_load(f)
                
                # Check if model matches a key in vllm_model_config.yaml
                # Support exact match or basename match (e.g. path/to/Qwen3-4B -> Qwen3-4B)
                target_model = None
                possible_keys = [model_name, os.path.basename(model_name)]
                
                for key in possible_keys:
                    if key in model_config.get("models", {}):
                        target_model = key
                        break
                
                if not target_model:
                     raise ValueError(f"[LLM Wrapper] Model '{model_name}' not found in {config_path}. Available models: {list(model_config.get('models', {}).keys())}")

                if target_model:
                    # Update model_name to the short name (key in yaml) so it matches what vLLM is serving
                    # vLLM is started with --served-model-name <short_name>
                    if model_name != target_model:
                        logger.info(f"[LLM Wrapper] Auto-mapping vLLM Model Name: {model_name} -> {target_model}")
                        model_name = target_model

                    m_info = model_config["models"][target_model]
                    port = m_info.get("port")
                    host = m_info.get("host", "localhost")
                    
                    if not port:
                         raise ValueError(f"[LLM Wrapper] Port not configured for model '{target_model}' in {config_path}")

                    new_base_url = f"http://{host}:{port}/v1"
                    # Only log if it's changing the provided (or default) base_url
                    if base_url != new_base_url:
                        logger.info(f"[LLM Wrapper] Auto-configured vLLM Host/Port: {base_url} -> {new_base_url} for model {target_model}")
                        base_url = new_base_url
            except Exception as e:
                # Re-raise known errors, wrap others
                if isinstance(e, (FileNotFoundError, ValueError)):
                    raise e
                raise RuntimeError(f"[LLM Wrapper] Error parsing vLLM config: {e}")

            if not api_key:
                api_key = "EMPTY"

        self.cfg = LLMConfig(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout=int(timeout),
            max_retries=int(max_retries),
            debug=bool(debug),
            backend=backend,
            device_map=device_map,
            dtype=dtype
        )
        self.belief_dim = belief_dim

        if self.cfg.backend == "local":
            if model_manager is None:
                raise ImportError("Could not import model_manager. Check your python path.")
            logger.info(f"[LLM Wrapper] Initializing LOCAL backend: {model_name} on {device_map}")
            # Trigger pre-loading
            model_manager.load_model(model_name, device=device_map, dtype=dtype)
        elif self.cfg.backend == "vllm":
            logger.info(f"[LLM Wrapper] Initializing vLLM backend: {model_name} at {base_url}")
        else:
            masked = (api_key[:4] + "*" * max(0, len(api_key) - 8) + api_key[-4:]) if api_key else "(empty)"
            logger.info(f"[LLM Wrapper] Initializing API backend: {model_name} (Key: {masked})")

    # ---------------------- public API ----------------------
    def generate_response(self,
                          prompt: Any, # str or List[Dict]
                          temperature: Optional[float] = None,
                          top_p: Optional[float] = None,
                          repetition_penalty: Optional[float] = None,
                          max_tokens: Optional[int] = None,
                          stop: Optional[List[str]] = None) -> GenerationOutput:
        """
        Generate text using the configured backend.
        Returns GenerationOutput (subclass of str) with .usage metadata.
        """
        temp = float(self._pick(temperature, self.cfg.default_temperature))
        tp = float(self._pick(top_p, self.cfg.default_top_p))
        rp = float(self._pick(repetition_penalty, self.cfg.default_repetition_penalty))
        mt = int(self._pick(max_tokens, self.cfg.default_max_tokens))

        if self.cfg.backend == "local":
            return self._generate_local(prompt, temp, tp, rp, mt)
        else:
            # API & vLLM path
            # API might need string only, or check payload support
            if isinstance(prompt, list):
                # Simple conversion for API if it expects string in 'content'
                # Or pass list if 'messages' payload supports it.
                # For now, assume API 'prompt' arg maps to user content.
                # But wait, API payload constructs messages=[{role: user, content: prompt}]
                # If prompt is already a list, we should use it as messages directly.
                return self._generate_api_messages(prompt, temp, tp, rp, mt, stop)
            return self._generate_api(prompt, temp, tp, rp, mt, stop)

    def generate_batch(self,
                       prompts: List[Any],
                       temperatures: List[float],
                       repetition_penalties: List[float],
                       max_tokens: Optional[int] = None,
                       top_p: Optional[float] = None,
                       top_ps: Optional[List[float]] = None) -> List[str]:
        """
        Batch generation interface.
        Supports heterogeneous top_ps if provided.
        """
        N = len(prompts)
        if N == 0:
            return []
        
        # Defaults
        mt = int(self._pick(max_tokens, self.cfg.default_max_tokens))
        
        # If global top_p is provided but not top_ps, fill top_ps
        default_tp = float(self._pick(top_p, self.cfg.default_top_p))

        # Normalize params lists
        def _fill(lst, default_val):
            if lst is None: return [default_val] * N
            if len(lst) < N: return lst + [default_val] * (N - len(lst))
            return lst[:N]

        temps = _fill(temperatures, self.cfg.default_temperature)
        reps = _fill(repetition_penalties, self.cfg.default_repetition_penalty)
        
        # Fill top_ps
        final_top_ps = _fill(top_ps, default_tp)

        if self.cfg.backend == "local":
            try:
                return model_manager.generate_batch(
                    model_path=self.cfg.model_name,
                    prompts=prompts,
                    temperatures=temps,
                    repetition_penalties=reps,
                    max_new_tokens=mt,
                    top_ps=final_top_ps # Pass list to manager
                )
            except Exception as e:
                logger.error(f"[LLM Local Batch] Failed: {e}")
                return [""] * N
        else:
            # API & vLLM Backend: Parallel execution
            import concurrent.futures
            results = [""] * N
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(N, 10)) as executor:
                futures = {}
                for i in range(N):
                    # Reuse single generate logic
                    # Note: self.generate_response handles prompt formatting (str vs list) internally
                    f = executor.submit(
                        self.generate_response, 
                        prompt=prompts[i],
                        temperature=temps[i],
                        repetition_penalty=reps[i],
                        max_tokens=mt,
                        top_p=final_top_ps[i] # Use individual top_p
                    )
                    futures[f] = i
                
                for f in concurrent.futures.as_completed(futures):
                    idx = futures[f]
                    try:
                        results[idx] = f.result()
                    except Exception as e:
                        logger.error(f"[LLM API Batch] Request {idx} failed: {e}")
            return results

    def _generate_local(self, prompt: Any, temperature: float, top_p: float, 
                        repetition_penalty: float, max_tokens: int) -> GenerationOutput:
        try:
            # If prompt is string, wrap it for chat template consistency
            final_prompt = prompt
            if isinstance(prompt, str):
                final_prompt = [{"role": "user", "content": prompt}]

            text = model_manager.generate(
                model_path=self.cfg.model_name,
                prompt=final_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            # Estimate usage (naive split)
            p_tokens = sum(len(m['content'].split()) for m in final_prompt) * 1.3
            c_tokens = len(text.split()) * 1.3
            return self._postprocess_text(text, usage={"prompt_tokens": int(p_tokens), "completion_tokens": int(c_tokens)})
        except Exception as e:
            logger.error(f"[LLM Local] Generation failed: {e}")
            return GenerationOutput("")

    def _generate_api(self, prompt: str, temperature: float, top_p: float,
                      repetition_penalty: float, max_tokens: int, stop: Optional[List[str]]) -> GenerationOutput:
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stream": False
        }
        if self.cfg.backend == "vllm":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        if stop:
            payload["stop"] = stop
        return self._execute_api_request(url, headers, payload)

    def _generate_api_messages(self, messages: List[Dict[str, str]], temperature: float, top_p: float,
                               repetition_penalty: float, max_tokens: int, stop: Optional[List[str]]) -> GenerationOutput:
        # New helper for pre-formatted messages
        url = f"{self.cfg.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "repetition_penalty": repetition_penalty,
            "stream": False
        }
        if self.cfg.backend == "vllm":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        
        if stop:
            payload["stop"] = stop
        return self._execute_api_request(url, headers, payload)

    def _execute_api_request(self, url, headers, payload) -> GenerationOutput:
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout)
                if resp.status_code != 200:
                    if self.cfg.debug:
                        logger.warning(f"[LLM] HTTP {resp.status_code}: {resp.text}")
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(min(1.5 * (attempt + 1), 6.0))
                    continue
                data = resp.json()
                text = (data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")).strip()
                usage = data.get("usage", {})
                return self._postprocess_text(text, usage=usage)
            except Exception as e:
                last_err = e
                if self.cfg.debug:
                    logger.warning(f"[LLM] request failed (attempt {attempt+1}/{self.cfg.max_retries}): {e}")
                time.sleep(min(1.5 * (attempt + 1), 6.0))

        logger.error(f"[LLM] All retries failed, fallback empty output. Last error: {last_err}")
        return GenerationOutput("")

    # ---------------------- helpers ----------------------
    def _pick(self, val, default):
        return default if val is None else val

    def _postprocess_text(self, s: str, usage: Dict[str, int] = None) -> GenerationOutput:
        """
        Fix critical control characters & normalize lines.
        Key fix: JSON-decoded \b -> backspace(0x08). Replace with literal \b so '\boxed' survives.
        """
        if s is None:
            return GenerationOutput("")
        # Replace control chars that could mutate content
        # \b(backspace)=\x08, \f(formfeed)=\x0c
        s = s.replace("\x08", "\\b")  # critical for \boxed
        s = s.replace("\x0c", "\\f")
        # normalize CR/LF
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # unicode separators
        s = s.replace("\u2028", " ").replace("\u2029", " ")
        return GenerationOutput(s, usage=usage)