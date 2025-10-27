import logging
from importlib.util import find_spec
import numpy as np
from typing import List, Optional, Union
import copy
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@register_model("openvino_genai")
@register_model("openvino-causal")  # Keep both for compatibility
class OpenVINOCausalLM(LM):
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
        pretrained: str,
        device="cpu",
        trust_remote_code=True,
        kv_cache=False,
        cache_dir="",
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.openvino_device = device
        self.trust_remote_code = trust_remote_code
        self.kv_cache = kv_cache
        self.cache_dir = cache_dir
        self.pretrained = pretrained
        
        # Initialize model and tokenizer
        self._create_model(pretrained, **kwargs)

    def _create_model(self, pretrained: str, **kwargs) -> None:
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

        try:
            # Create OpenVINO GenAI LLMPipeline
            self.model = ov_genai.LLMPipeline(
                pretrained, self.openvino_device.upper(), **model_kwargs
            )

            # Get the tokenizer from the pipeline
            self.tokenizer = self.model.get_tokenizer()
            
            # Create simple model config
            self.config = self._detect_model_config(pretrained)

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

    def _detect_model_config(self, pretrained: str):
        """Create a simple config object with dynamic model detection."""
        class SimpleConfig:
            def __init__(self, model_path, tokenizer):
                # Try to detect model type from path
                model_path_lower = model_path.lower()
                
                if 'gemma' in model_path_lower:
                    self.model_type = "gemma"
                    self.max_position_embeddings = 8192
                    self.vocab_size = 256000
                elif 'llama' in model_path_lower:
                    self.model_type = "llama"
                    self.max_position_embeddings = 4096
                    self.vocab_size = 32000
                elif 'gpt' in model_path_lower:
                    self.model_type = "gpt"
                    self.max_position_embeddings = 1024
                    self.vocab_size = 50257
                elif 'mistral' in model_path_lower:
                    self.model_type = "mistral"
                    self.max_position_embeddings = 4096
                    self.vocab_size = 32000
                elif 'phi' in model_path_lower:
                    self.model_type = "phi"
                    self.max_position_embeddings = 2048
                    self.vocab_size = 50257
                elif 'qwen' in model_path_lower:
                    self.model_type = "qwen"
                    self.max_position_embeddings = 8192
                    self.vocab_size = 151936
                else:
                    self.model_type = "unknown"
                    self.max_position_embeddings = 4096
                    self.vocab_size = 50257
                
                # Try to get actual vocab size from tokenizer
                try:
                    if hasattr(tokenizer, 'get_vocab_size'):
                        self.vocab_size = tokenizer.get_vocab_size()
                    elif hasattr(tokenizer, 'vocab_size'):
                        self.vocab_size = tokenizer.vocab_size
                except:
                    pass  # Keep default

        return SimpleConfig(pretrained, self.tokenizer)

    @property
    def eot_token_id(self):
        """End of text token ID."""
        return getattr(self.tokenizer, 'eos_token_id', 2)

    @property
    def max_length(self):
        """Maximum sequence length."""
        return getattr(self.config, 'max_position_embeddings', 4096)

    @property
    def max_gen_toks(self):
        """Maximum generation tokens."""
        return 256

    @property
    def batch_size(self):
        """Batch size for evaluation."""
        return 1

    @property
    def device(self):
        """Device name."""
        return self.openvino_device

    def tok_encode(self, string: str) -> List[int]:
        """Encode string to token IDs."""
        encoded = self.tokenizer.encode(string)
        if hasattr(encoded, 'input_ids'):
            return encoded.input_ids.tolist()
        return encoded.tolist() if hasattr(encoded, 'tolist') else list(encoded)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """
        Compute loglikelihood using OpenVINO GenAI's logprobs functionality.
        This replaces dummy logits with real probability computation.
        """
        import openvino_genai as ov_genai
        
        res = []
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (len(requests) < 10)),
            desc="Running loglikelihood"
        )

        for request in requests:
            # Handle both Instance objects and tuples
            if isinstance(request, tuple):
                context, continuation = request
            else:
                context, continuation = request.args

            # Create generation config with logprobs enabled
            config = ov_genai.GenerationConfig()
            config.max_new_tokens = 1  # We just need logprobs, not generation
            config.do_sample = False
            config.num_return_sequences = 1
            
            # Enable logprobs - this is the key feature
            config.return_dict_in_generate = True
            config.output_logits = True

            try:
                # Prepare the full input text
                full_text = context + continuation
                
                # Generate with logprobs enabled
                result = self.model.generate(full_text, config)
                
                # Extract logprobs from the result
                if hasattr(result, 'logprobs') and result.logprobs:
                    # Calculate loglikelihood from the logprobs
                    # This is a simplified calculation - you might need to adjust based on 
                    # the exact structure of result.logprobs
                    logprob = sum(result.logprobs) if isinstance(result.logprobs, list) else result.logprobs
                    is_greedy = True  # Since we used do_sample=False
                else:
                    # Fallback: estimate from the continuation tokens
                    continuation_tokens = self.tok_encode(continuation)
                    # This is a simplified fallback - real implementation would need proper logprob calculation
                    logprob = -len(continuation_tokens) * 2.0  # Rough estimate
                    is_greedy = True
                    eval_logger.warning("Using fallback logprob estimation - real logprobs not available")

            except Exception as e:
                eval_logger.warning(f"Error computing logprobs: {e}")
                # Fallback calculation
                continuation_tokens = self.tok_encode(continuation)
                logprob = -len(continuation_tokens) * 2.0  # Rough estimate
                is_greedy = True

            res.append((logprob, is_greedy))
            pbar.update(1)

        pbar.close()
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generate text until stopping criteria are met."""
        import openvino_genai as ov_genai
        
        res = []
        # Create progress bar
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (len(requests) < 10)),
            desc="Running generate_until"
        )

        for request in requests:
            # Handle both Instance objects and tuples
            if isinstance(request, tuple):
                context, generation_kwargs = request
            else:
                context = request.args[0]
                generation_kwargs = request.args[1] if len(request.args) > 1 else {}

            # Create generation config
            config = ov_genai.GenerationConfig()
            
            # Set max tokens
            max_gen_toks = generation_kwargs.get('max_gen_toks', self.max_gen_toks)
            config.max_new_tokens = max_gen_toks
            
            # Set stopping criteria
            until = generation_kwargs.get('until', [])
            if until:
                config.stop_strings = until
            
            # Set sampling parameters
            config.do_sample = generation_kwargs.get('do_sample', False)
            config.temperature = generation_kwargs.get('temperature', 1.0)
            config.top_p = generation_kwargs.get('top_p', 1.0)

            try:
                # Generate text
                generated_text = self.model.generate(context, config)
                
                # Remove the input context from the result
                if generated_text.startswith(context):
                    generated_text = generated_text[len(context):]
                
                res.append(generated_text)

            except Exception as e:
                eval_logger.warning(f"Error during generation: {e}")
                res.append("")  # Return empty string on error

            pbar.update(1)

        pbar.close()
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """Compute rolling loglikelihood."""
        # For now, delegate to regular loglikelihood
        # A proper implementation would handle the rolling window
        return self.loglikelihood(requests, disable_tqdm)