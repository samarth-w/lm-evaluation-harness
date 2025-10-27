import logging
from importlib.util import find_spec

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)


@register_model("openvino-causal")
class OpenVINOCausalLM(HFLM):
    """
    OpenVINO GenAI provides a simple interface to run generative AI models optimized for
    Intel® architectures using OpenVINO™ runtime with built-in performance optimizations.

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
                pretrained,
                self.openvino_device.upper(),
                **ov_properties
            )

            # Get the tokenizer from the pipeline
            self.tokenizer = self._model.get_tokenizer()

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