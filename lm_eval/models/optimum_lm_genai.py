import logging
from importlib.util import find_spec

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)


@register_model("openvino-causal")
class OpenVINOCausalLM(HFLM):
    """
    OpenVINO GenAI provides a simple interface to run generative AI models optimized for
    Intel architectures using OpenVINO runtime with built-in performance optimizations.

    Example usage:
    `lm_eval --model openvino-causal --model_args pretrained=gpt2,device=cpu,kv_cache=true --task wikitext`

    This implementation uses OpenVINO GenAI library directly, avoiding external dependencies
    like llm_bench as suggested in PR #1862 comments.
    """

    def __init__(
        self,
        device="cpu",
        convert_tokenizer=False,
        trust_remote_code=True,
        kv_cache=False,
        **kwargs,
    ) -> None:
        self.openvino_device = device
        self.trust_remote_code = trust_remote_code
        self.convert_tokenizer = convert_tokenizer
        self.kv_cache = kv_cache

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
            "CACHE_DIR": "",
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
                
                # Try to get vocab size
                try:
                    self.vocab_size = getattr(ov_tokenizer, 'get_vocab_size', lambda: 32000)()
                except:
                    self.vocab_size = 32000
                    eval_logger.warning("Could not determine vocab size, using default: 32000")
                
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
            
            def __getattr__(self, name):
                # Delegate all other attributes/methods to the original tokenizer
                return getattr(self._tokenizer, name)
        
        # Replace the tokenizer with our wrapper
        self.tokenizer = TokenizerWrapper(self.tokenizer)
