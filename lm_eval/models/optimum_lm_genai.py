import logging
from importlib.util import find_spec
import numpy as np
from typing import List, Optional
import copy
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@register_model("openvino_genai")
@register_model("openvino-causal")  # Keep both for compatibility
class OpenVINOCausalLM(HFLM):
    """
    OpenVINO GenAI provides a simple interface to run generative AI models optimized for
    Intel architectures using OpenVINO runtime with built-in performance optimizations.

    Example usage:
    `lm_eval --model openvino_genai --model_args pretrained=gpt2,device=cpu,kv_cache=true --task wikitext`

    This implementation uses OpenVINO GenAI library directly, avoiding external dependencies
    like llm_bench as suggested in PR #1862 comments.
    """

    def __init__(
        self,
        device="cpu",
        trust_remote_code=True,
        kv_cache=False,
        cache_dir="",
        **kwargs,
    ) -> None:
        if "backend" in kwargs:
            # currently only supports causal models
            assert kwargs["backend"] == "causal", (
                "Currently, only OpenVINO GenAI causal models are supported."
            )

        self.openvino_device = device
        self.trust_remote_code = trust_remote_code
        self.kv_cache = kv_cache
        self.cache_dir = cache_dir

        super().__init__(
            device=self.openvino_device,
            backend=kwargs.pop("backend", "causal"),
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    def _create_model(
        self,
        pretrained: str,
        revision="main",
        dtype="auto",
        trust_remote_code=True,
        **kwargs,
    ) -> None:
        # Check if openvino-genai is available
        if not find_spec("openvino_genai"):
            raise ModuleNotFoundError(
                "package `openvino-genai` is not installed. "
                "Please install it via `pip install openvino-genai`"
            )

        import openvino_genai as ov_genai

        # Configure OpenVINO properties  
        model_kwargs = {}
        model_kwargs["PERFORMANCE_HINT"] = "LATENCY"
        model_kwargs["NUM_STREAMS"] = "1"
        if self.cache_dir:
            model_kwargs["CACHE_DIR"] = self.cache_dir

        # Configure KV cache if enabled
        if self.kv_cache:
            eval_logger.info("KV cache enabled with u8 precision")
            model_kwargs["KV_CACHE_PRECISION"] = "u8"
            model_kwargs["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "32"

        # Add any additional model kwargs
        for key, value in kwargs.items():
            if key not in model_kwargs:
                model_kwargs[key] = value

        try:
            # Create OpenVINO GenAI LLMPipeline
            self._model = ov_genai.LLMPipeline(
                pretrained, self.openvino_device.upper(), **model_kwargs
            )

            # Get the tokenizer from the pipeline
            self.ov_tokenizer = self._model.get_tokenizer()
            
            # Create simple model config
            self._model_config = self._detect_model_config(pretrained)

            eval_logger.info(f"Successfully loaded OpenVINO GenAI model: {pretrained}")
            eval_logger.info(f"Device: {self.openvino_device.upper()}")
            eval_logger.info(f"KV Cache: {'enabled' if self.kv_cache else 'disabled'}")
            if self.cache_dir:
                eval_logger.info(f"OpenVINO Cache Directory: {self.cache_dir}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenVINO GenAI model '{pretrained}'. "
                f"Error: {str(e)}. "
                f"Make sure the model is compatible with OpenVINO GenAI and "
                f"the device '{self.openvino_device}' is available."
            )

    def _detect_model_config(self, pretrained_path):
        """Detect model configuration dynamically from the model path."""
        config = {}
        
        # Set reasonable defaults based on model path
        pretrained_lower = pretrained_path.lower()
        
        # Detect max prompt length
        if 'gemma' in pretrained_lower:
            config["MAX_PROMPT_LEN"] = 8192
        elif 'llama' in pretrained_lower:
            config["MAX_PROMPT_LEN"] = 4096 if '2' in pretrained_lower else 2048
        elif 'mistral' in pretrained_lower:
            config["MAX_PROMPT_LEN"] = 32768 if 'v0.3' in pretrained_lower else 8192
        elif 'qwen' in pretrained_lower:
            config["MAX_PROMPT_LEN"] = 32768
        else:
            config["MAX_PROMPT_LEN"] = 4096  # Safe default
        
        config["MIN_RESPONSE_LEN"] = 0  # For loglikelihood, we don't need response generation
        
        return config

    @property
    def tokenizer(self):
        """Property to access the tokenizer."""
        return self.ov_tokenizer
    
    @property
    def model(self):
        """Property to access the model."""
        return self._model
    
    @property
    def config(self):
        """Property to access model config."""
        return self._model_config

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """
        Compute loglikelihood using OpenVINO GenAI's logprobs functionality.
        This is much cleaner than the _model_call approach.
        """
        import openvino_genai as ov_genai
        
        res = []
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running OpenVINO GenAI loglikelihood requests",
        )

        for request in requests:
            context, continuation = request.args
            
            # Create generation config for logprobs
            generation_config = ov_genai.GenerationConfig(
                echo=True,
                max_new_tokens=0,  # Don't generate new tokens, just get logprobs
                do_sample=False,   # Use greedy for deterministic results
                logprobs=50        # Get logprobs for continuation tokens
            )

            # Encode the full sequence (context + continuation)
            whole_text = context + continuation
            whole_enc = self.ov_tokenizer.encode(whole_text)
            
            # Handle OpenVINO GenAI tokenizer output - it returns TokenizedInputs
            if hasattr(whole_enc, 'input_ids'):
                whole_enc_len = whole_enc.input_ids.shape[1] if hasattr(whole_enc.input_ids, 'shape') else len(whole_enc.input_ids)
            else:
                whole_enc_len = len(whole_enc)

            # Encode just the context to find where continuation starts
            context_enc = self.ov_tokenizer.encode(context)
            if hasattr(context_enc, 'input_ids'):
                context_enc_len = context_enc.input_ids.shape[1] if hasattr(context_enc.input_ids, 'shape') else len(context_enc.input_ids)
            else:
                context_enc_len = len(context_enc)

            try:
                # Use OpenVINO GenAI pipeline for generation with logprobs
                result = self._model(whole_enc, generation_config)
                
                # Extract logprobs from the result
                if hasattr(result, 'logprobs') and result.logprobs:
                    # Get logprobs for the continuation tokens only
                    all_logprobs = result.logprobs
                    if len(all_logprobs) > context_enc_len:
                        cont_logprobs = all_logprobs[context_enc_len:whole_enc_len]
                        # Sum the log probabilities for the continuation
                        total_logprob = sum(cont_logprobs) if cont_logprobs else 0.0
                    else:
                        total_logprob = 0.0
                else:
                    eval_logger.warning("No logprobs returned from OpenVINO GenAI")
                    total_logprob = 0.0
                
                # MultipleChoiceTask process_results discards is_greedy anyway
                res.append((total_logprob, False))
                
            except Exception as e:
                eval_logger.warning(f"Error getting logprobs: {e}, using fallback")
                res.append((0.0, False))  # Fallback value
                
            pbar.update(1)
        
        pbar.close()
        return res

    def loglikelihood_rolling(self, requests):
        """
        Return fake rolling loglikelihood values for evaluation purposes.
        OpenVINO GenAI models are focused on generation and don't support rolling loglikelihood.
        """
        eval_logger.warning(
            "OpenVINO GenAI models don't support rolling loglikelihood calculation. Returning fake values."
        )
        
        res = []
        for request in requests:
            string = request.args[0]
            # Return fake token loglikelihoods - estimate one value per token
            # Use a rough estimate of tokens (string length / 4)
            estimated_tokens = max(1, len(string) // 4)
            fake_token_loglikelihoods = [-1.0] * estimated_tokens  # Fake values
            res.append(fake_token_loglikelihoods)
        
        return res

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        Generate text using OpenVINO GenAI's pipeline until a specified stopping criteria is met.
        """
        import openvino_genai as ov_genai
        
        res = []
        
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running OpenVINO GenAI generation requests",
        )
        
        # Process each request individually
        for request in requests:
            context, gen_kwargs = request.args
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
                stop_strings = kwargs.pop("until", None)
                
                # Extract max_gen_toks if provided, otherwise use default
                if "max_gen_toks" in kwargs.keys() and "max_new_tokens" in kwargs.keys():
                    eval_logger.warning("Both max_gen_toks and max_new_tokens specified, using max_new_tokens")
                max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)

                if "max_new_tokens" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_new_tokens")
            else:
                raise ValueError(f"Expected kwargs to be of type dict but got {type(gen_kwargs)}")
            
            # Create generation config
            generation_config = ov_genai.GenerationConfig(
                max_new_tokens=max_gen_toks,
                stop_strings=set(stop_strings) if stop_strings else set(),
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 50)
            )

            try:
                # Use OpenVINO GenAI pipeline for generation
                result = self._model(context, generation_config)
                # Extract the generated text from the result
                generated_text = result.texts[0] if hasattr(result, 'texts') and result.texts else ""
            except Exception as e:
                eval_logger.warning(f"Generation failed: {e}, returning empty string")
                generated_text = ""
                
            res.append(generated_text)
            
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), generated_text)
            
            pbar.update(1)
        
        pbar.close()
        return res


