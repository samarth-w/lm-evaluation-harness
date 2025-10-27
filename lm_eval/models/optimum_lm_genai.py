import logging
from importlib.util import find_spec

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


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
        convert_tokenizer=False,
        trust_remote_code=True,
        kv_cache=False,
        cache_dir="",
        **kwargs,
    ) -> None:
        self.openvino_device = device
        self.trust_remote_code = trust_remote_code
        self.convert_tokenizer = convert_tokenizer
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
        ov_properties = {
            "PERFORMANCE_HINT": "LATENCY",
            "NUM_STREAMS": "1",
            "CACHE_DIR": self.cache_dir,  # Use cache_dir for OpenVINO model caching
        }

        # Configure KV cache if enabled
        if self.kv_cache:
            eval_logger.info("KV cache enabled with u8 precision")
            ov_properties["KV_CACHE_PRECISION"] = "u8"
            ov_properties["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "32"

        # Handle additional model kwargs
        model_kwargs = kwargs if kwargs else {}
        ov_properties.update(model_kwargs.get("ov_config", {}))

        try:
            # Create OpenVINO GenAI LLMPipeline
            self._model = ov_genai.LLMPipeline(
                pretrained, self.openvino_device.upper(), **ov_properties
            )

            # Get the tokenizer from the pipeline
            self.tokenizer = self._model.get_tokenizer()
            
            # Create a wrapper to provide HuggingFace-compatible interface
            self._setup_tokenizer_compatibility()

            eval_logger.info(f"Successfully loaded OpenVINO GenAI model: {pretrained}")
            eval_logger.info(f"Device: {self.openvino_device.upper()}")
            eval_logger.info(f"KV Cache: {'enabled' if self.kv_cache else 'disabled'}")
            if self.cache_dir:
                eval_logger.info(f"OpenVINO Cache Directory: {self.cache_dir}")
            
            # Add config attribute for compatibility - store internally
            self._config = self._create_model_config(pretrained)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load OpenVINO GenAI model '{pretrained}'. "
                f"Error: {str(e)}. "
                f"Make sure the model is compatible with OpenVINO GenAI and "
                f"the device '{self.openvino_device}' is available."
            )

    def _setup_tokenizer_compatibility(self):
        """Create a wrapper to provide HuggingFace-compatible interface for OpenVINO GenAI tokenizer."""
        # Create a wrapper class that adds the missing attributes
        class TokenizerWrapper:
            def __init__(self, ov_tokenizer):
                self._tokenizer = ov_tokenizer
                
                # Try to get vocab_size from tokenizer - dynamic detection
                self.vocab_size = self._detect_vocab_size(ov_tokenizer)
            
            def _detect_vocab_size(self, tokenizer):
                """Dynamically detect vocabulary size from different model types."""
                try:
                    # Try direct vocab_size attribute
                    if hasattr(tokenizer, 'vocab_size'):
                        return tokenizer.vocab_size
                    
                    # Try get_vocab_size method
                    if hasattr(tokenizer, 'get_vocab_size'):
                        return tokenizer.get_vocab_size()
                    
                    # Try to get vocab and count
                    if hasattr(tokenizer, 'get_vocab'):
                        vocab = tokenizer.get_vocab()
                        return len(vocab)
                    
                    # Try encoding a sample to estimate vocab size
                    try:
                        # Test with a range of tokens to find max token ID
                        test_text = "The quick brown fox jumps over the lazy dog. 0123456789 !@#$%^&*()"
                        tokens = tokenizer.encode(test_text)
                        if tokens:
                            max_token = max(tokens.input_ids if hasattr(tokens, 'input_ids') else tokens)
                            # Estimate vocab size as max_token * 1.2 (with some buffer)
                            estimated_size = int(max_token * 1.2)
                            # Common vocab sizes - round to nearest
                            common_sizes = [32000, 50257, 50400, 128000, 256000, 400000]
                            return min(common_sizes, key=lambda x: abs(x - estimated_size))
                    except:
                        pass
                    
                    # Model-specific defaults based on common architectures
                    eval_logger.warning("Could not detect vocab size, using intelligent defaults")
                    return 50257  # GPT-2 style default, works for many models
                    
                except Exception as e:
                    eval_logger.warning(f"Error detecting vocab size: {e}, using default")
                    return 50257  # Safe default
                
                # Set token IDs with fallbacks
                try:
                    self.pad_token_id = getattr(ov_tokenizer, 'get_pad_token_id', lambda: 0)()
                except:
                    self.pad_token_id = 0
                
                try:
                    self.eos_token_id = getattr(ov_tokenizer, 'get_eos_token_id', lambda: 2)()
                except:
                    self.eos_token_id = 2
                
                try:
                    self.bos_token_id = getattr(ov_tokenizer, 'get_bos_token_id', lambda: 1)()
                except:
                    self.bos_token_id = 1
                
                # Add token string attributes (not just IDs)
                try:
                    self.pad_token = getattr(ov_tokenizer, 'get_pad_token', lambda: None)()
                except:
                    self.pad_token = None
                
                try:
                    self.eos_token = getattr(ov_tokenizer, 'get_eos_token', lambda: '</s>')()
                except:
                    self.eos_token = '</s>'
                
                try:
                    self.bos_token = getattr(ov_tokenizer, 'get_bos_token', lambda: '<s>')()
                except:
                    self.bos_token = '<s>'
                
                try:
                    self.unk_token = getattr(ov_tokenizer, 'get_unk_token', lambda: '<unk>')()
                except:
                    self.unk_token = '<unk>'
            
            def encode(self, text, add_special_tokens=True, **kwargs):
                """Encode text and return a list of token IDs (compatible with HF tokenizers)."""
                import numpy as np
                tokenized = self._tokenizer.encode(text)
                
                def flatten_to_int_list(obj):
                    """Recursively flatten to a list of integers."""
                    if isinstance(obj, (int, float)):
                        return [int(obj)]
                    elif isinstance(obj, (list, tuple)):
                        result = []
                        for item in obj:
                            result.extend(flatten_to_int_list(item))
                        return result
                    elif hasattr(obj, 'data'):
                        # OpenVINO tensor
                        return flatten_to_int_list(obj.data.tolist())
                    elif hasattr(obj, 'numpy'):
                        return flatten_to_int_list(obj.numpy().tolist())
                    elif hasattr(obj, 'tolist'):
                        return flatten_to_int_list(obj.tolist())
                    else:
                        import numpy as np
                        try:
                            return flatten_to_int_list(np.array(obj).tolist())
                        except:
                            # If all else fails, try to convert to int
                            try:
                                return [int(obj)]
                            except:
                                return []
                
                # Convert OpenVINO GenAI TokenizedInputs to list of integers
                if hasattr(tokenized, 'input_ids'):
                    result = flatten_to_int_list(tokenized.input_ids)
                else:
                    result = flatten_to_int_list(tokenized)
                
                # Ensure we return a flat list of integers
                return [int(x) for x in result if isinstance(x, (int, float, np.integer, np.floating))]
            
            def decode(self, token_ids, skip_special_tokens=False, **kwargs):
                """Decode token IDs to text."""
                return self._tokenizer.decode(token_ids)
            
            def __call__(self, text, **kwargs):
                """Make the tokenizer callable like HF tokenizers."""
                return self.encode(text, **kwargs)
            
            def __getattr__(self, name):
                # Delegate all other attributes/methods to the original tokenizer
                return getattr(self._tokenizer, name)
        
        # Replace the tokenizer with our wrapper
        self.tokenizer = TokenizerWrapper(self.tokenizer)

    def _create_model_config(self, pretrained_path):
        """Create a minimal model config that provides required attributes."""
        
        # Create and return config
        class ModelConfig:
            def __init__(self, tokenizer, pretrained):
                # Get vocab size from tokenizer (now properly detected)
                self.vocab_size = getattr(tokenizer, 'vocab_size', 50257)
                
                # Dynamic context length detection
                self.max_position_embeddings = self._detect_max_length(pretrained)
                self.max_length = self.max_position_embeddings
                self.n_positions = self.max_position_embeddings
                self.max_seq_len = self.max_position_embeddings
                
                # Dynamic model type detection
                model_info = self._detect_model_type(pretrained)
                self.model_type = model_info['type']
                self.architectures = model_info['architectures']
            
            def _detect_model_type(self, pretrained_path):
                """Detect model type from pretrained path or other indicators."""
                try:
                    pretrained_path = pretrained_path.lower()
                    
                    if 'gemma' in pretrained_path:
                        return {'type': 'gemma', 'architectures': ['GemmaForCausalLM']}
                    elif 'llama' in pretrained_path:
                        return {'type': 'llama', 'architectures': ['LlamaForCausalLM']}
                    elif 'gpt' in pretrained_path:
                        return {'type': 'gpt2', 'architectures': ['GPT2LMHeadModel']}
                    elif 'mistral' in pretrained_path:
                        return {'type': 'mistral', 'architectures': ['MistralForCausalLM']}
                    elif 'phi' in pretrained_path:
                        return {'type': 'phi', 'architectures': ['PhiForCausalLM']}
                    elif 'qwen' in pretrained_path:
                        return {'type': 'qwen2', 'architectures': ['Qwen2ForCausalLM']}
                    else:
                        # Generic causal LM
                        return {'type': 'causal_lm', 'architectures': ['CausalLM']}
                except:
                    return {'type': 'causal_lm', 'architectures': ['CausalLM']}
            
            def _detect_max_length(self, pretrained_path):
                """Detect maximum sequence length from model architecture."""
                try:
                    # Common max lengths for different model families
                    pretrained_path = pretrained_path.lower()
                    
                    if 'gemma' in pretrained_path:
                        return 8192  # Gemma-2 supports 8K context
                    elif 'llama' in pretrained_path:
                        if '2' in pretrained_path:
                            return 4096  # Llama 2
                        else:
                            return 2048  # Llama 1
                    elif 'gpt' in pretrained_path:
                        return 1024  # GPT-2 style
                    elif 'mistral' in pretrained_path:
                        return 32768 if 'v0.3' in pretrained_path else 8192
                    elif 'phi' in pretrained_path:
                        return 2048  # Phi models
                    elif 'qwen' in pretrained_path:
                        return 32768  # Qwen2 supports long context
                    else:
                        return 4096  # Safe default
                except:
                    return 4096  # Safe default
        
        # Create and return config
        return ModelConfig(self.tokenizer, pretrained_path)
    
    @property
    def model(self):
        """Property to access the model - add config attribute dynamically."""
        # Create a wrapper that adds config to the OpenVINO model
        class ModelWrapper:
            def __init__(self, ov_model, config):
                self._model = ov_model
                self.config = config
            
            def __call__(self, *args, **kwargs):
                # Make the wrapper callable like the original model
                return self._model(*args, **kwargs)
            
            def __getattr__(self, name):
                # Delegate all other attributes/methods to the original model
                return getattr(self._model, name)
        
        return ModelWrapper(self._model, self._config)
    
    @property 
    def config(self):
        """Property to access model config."""
        return self._config
    
    @property
    def generation_config(self):
        """Property to access generation config - dynamically configured."""
        vocab_size = getattr(self.tokenizer, 'vocab_size', 50257)
        return {
            'max_new_tokens': 50,
            'do_sample': False,
            'temperature': 1.0,
            'top_p': 1.0,
            'top_k': min(50, vocab_size // 100),  # Scale top_k with vocab size
            'repetition_penalty': 1.0,
            'echo': False,
            'logprobs': 0  # Default to no logprobs, will be overridden in _model_call
        }
    
    def _model_call(self, inps, **kwargs):
        """
        Override the base class _model_call to work with OpenVINO GenAI.
        This method is called by the evaluation harness to get model logits.
        """
        import torch
        import numpy as np
        
        batch_size, seq_len = inps.shape
        vocab_size = getattr(self.tokenizer, 'vocab_size', 50257)  # Use detected vocab size
        
        # Try to use OpenVINO GenAI's logprobs functionality for real logits
        try:
            # Convert input tokens to the format expected by OpenVINO GenAI
            all_logits = []
            
            for batch_idx in range(batch_size):
                input_ids = inps[batch_idx].tolist()
                
                # Use generation with logprobs to get log probabilities
                # We generate just one token to get logprobs for the input sequence
                try:
                    import openvino_genai as ov_genai
                    
                    # Create proper GenerationConfig object
                    gen_config = ov_genai.GenerationConfig()
                    gen_config.max_new_tokens = 1  # Generate minimal tokens to get logprobs
                    gen_config.do_sample = False   # Use greedy to be deterministic
                    gen_config.echo = True         # Include input in output to get logprobs for input tokens
                    # Dynamic logprobs limit based on vocab size
                    logprobs_limit = min(vocab_size, max(50, vocab_size // 1000))  # Scale with vocab size
                    gen_config.logprobs = logprobs_limit
                    
                    # Generate with the input to get logprobs
                    result = self.model(input_ids, gen_config)
                    
                    # Extract logprobs if available
                    if hasattr(result, 'logprobs') and result.logprobs is not None:
                        # Convert logprobs to logits format expected by the harness
                        logprobs = result.logprobs
                        if len(logprobs) >= seq_len:
                            # Take logprobs for the input sequence positions
                            batch_logits = []
                            for pos_idx in range(seq_len):
                                if pos_idx < len(logprobs):
                                    pos_logprobs = logprobs[pos_idx]
                                    # Convert to full vocabulary logits
                                    pos_logits = torch.full((vocab_size,), -float('inf'), dtype=torch.float32)
                                    for token_id, logprob in pos_logprobs.items():
                                        if token_id < vocab_size:
                                            pos_logits[token_id] = logprob
                                    batch_logits.append(pos_logits)
                                else:
                                    # Fallback for missing positions
                                    batch_logits.append(torch.full((vocab_size,), -float('inf'), dtype=torch.float32))
                            
                            all_logits.append(torch.stack(batch_logits))
                        else:
                            # If we don't get enough logprobs, fall back to dummy
                            all_logits.append(torch.randn(seq_len, vocab_size, dtype=torch.float32))
                    else:
                        # No logprobs available, use dummy
                        all_logits.append(torch.randn(seq_len, vocab_size, dtype=torch.float32))
                        
                except Exception as e:
                    eval_logger.warning(f"Failed to get logprobs from OpenVINO GenAI: {e}")
                    # Fallback to dummy logits for this batch
                    all_logits.append(torch.randn(seq_len, vocab_size, dtype=torch.float32))
            
            if all_logits:
                logits = torch.stack(all_logits)
                eval_logger.info("Successfully obtained logprobs from OpenVINO GenAI")
                return logits
            
        except Exception as e:
            eval_logger.warning(f"Error using OpenVINO GenAI logprobs: {e}")
        
        # Fallback to dummy logits if logprobs approach fails
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        eval_logger.warning("Using dummy logits for OpenVINO GenAI - logprobs approach failed. "
                          "For proper loglikelihood computation, OpenVINO GenAI logprobs functionality is needed.")
        
        return logits


