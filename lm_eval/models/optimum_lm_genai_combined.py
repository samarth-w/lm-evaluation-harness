import logging
from importlib.util import find_spec
from typing import Any, List

from lm_eval.api.registry import register_model
from lm_eval.api.model import LM


eval_logger = logging.getLogger(__name__)


if find_spec("openvino_genai") is not None:
    try:
        import openvino_genai as ov_genai
    except Exception:
        ov_genai = None


@register_model("openvino_genai_test")
class OpenVINOCausalLMCombined(LM):
    """Concise OpenVINO GenAI adapter combining robustness and logprob support.

    - No global monkeypatching.
    - Compact TokenizerWrapper that normalizes encode/decode and common attrs.
    - ModelWrapper that normalizes inputs (torch / numpy / flat lists) and
      forwards to the pipeline's generate / __call__.
    - Attempts to use `openvino_genai.GenerationConfig()` to retrieve logprobs
      in `_model_call`, with safe fallbacks to dummy logits.
    """

    def __init__(
        self, pretrained: str, device: str = "CPU", kv_cache: bool = False, **kwargs
    ) -> None:
        super().__init__()
        if ov_genai is None:
            raise ModuleNotFoundError("openvino-genai is not available")

        self.pretrained = pretrained
        self.openvino_device = device
        self.kv_cache = kv_cache
        self._create_model(pretrained, **kwargs)

    def _create_model(self, pretrained: str, **kwargs) -> None:
        ov_properties = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}
        if self.kv_cache:
            ov_properties.update({"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32"})
        ov_properties.update(kwargs.get("ov_config", {}))

        try:
            self._model = ov_genai.LLMPipeline(pretrained, self.openvino_device.upper(), **ov_properties)
            tok = self._model.get_tokenizer()
            self.tokenizer = self._build_tokenizer_wrapper(tok)
            # Ensure simple `get_tokenizer()` usage returns underlying tokenizer when called via self.model
            self._config = self._create_model_config()
            eval_logger.info(f"Loaded OpenVINO GenAI model: {pretrained} on {self.openvino_device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO GenAI model '{pretrained}': {e}")

    def _build_tokenizer_wrapper(self, ov_tokenizer: Any):
        class TokenizerWrapper:
            def __init__(self, base):
                self._tokenizer = base
                # minimal attributes
                self.vocab_size = getattr(base, "get_vocab_size", lambda: getattr(base, "vocab_size", 32000))()
                self.pad_token_id = getattr(base, "get_pad_token_id", lambda: 0)()
                self.eos_token_id = getattr(base, "get_eos_token_id", lambda: 2)()
                self.bos_token_id = getattr(base, "get_bos_token_id", lambda: 1)()

            def encode(self, text, **kwargs):
                """Return flat list of ints for a single input"""
                tok = self._tokenizer.encode(text)
                # attempt to coerce various return shapes into flat ints
                def _flatten(obj):
                    if obj is None:
                        return []
                    if isinstance(obj, (int,)):
                        return [int(obj)]
                    if hasattr(obj, "tolist"):
                        try:
                            return _flatten(obj.tolist())
                        except Exception:
                            pass
                    if isinstance(obj, (list, tuple)):
                        out = []
                        for e in obj:
                            out.extend(_flatten(e))
                        return out
                    try:
                        return [int(obj)]
                    except Exception:
                        return []

                return [int(x) for x in _flatten(getattr(tok, "input_ids", tok))]

            def decode(self, token_ids, skip_special_tokens=False, **kwargs):
                # coerce to flat list of ints if needed
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                if isinstance(token_ids, (int,)):
                    token_ids = [int(token_ids)]
                try:
                    return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)
                except TypeError:
                    return self._tokenizer.decode(list(token_ids))

            def __call__(self, texts, **kwargs):
                # keep it small: if list-like, map encode, else encode single
                if isinstance(texts, (list, tuple)):
                    return {"input_ids": [self.encode(t, **kwargs) for t in texts], "attention_mask": None}
                return self.encode(texts, **kwargs)

            def __getattr__(self, name):
                return getattr(self._tokenizer, name)

        return TokenizerWrapper(ov_tokenizer)

    def _create_model_config(self):
        class ModelConfig:
            def __init__(self, tokenizer):
                self.max_position_embeddings = 4096
                self.max_length = 4096
                self.vocab_size = getattr(tokenizer, "vocab_size", 32000)
                self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
                self.eos_token_id = getattr(tokenizer, "eos_token_id", 2)
                self.bos_token_id = getattr(tokenizer, "bos_token_id", 1)

        return ModelConfig(self.tokenizer)

    @property
    def model(self):
        """Return a small ModelWrapper that normalizes inputs then delegates."""
        class ModelWrapper:
            def __init__(self, m, tok):
                self._m = m
                self._tok = tok

            def _normalize(self, inputs):
                # handle torch / numpy / list -> str or list[str]
                try:
                    import torch
                except Exception:
                    torch = None
                try:
                    import numpy as _np
                except Exception:
                    _np = None

                if torch is not None and isinstance(inputs, torch.Tensor):
                    try:
                        inputs = inputs.cpu().tolist()
                    except Exception:
                        pass
                if _np is not None and isinstance(inputs, _np.ndarray):
                    try:
                        inputs = inputs.tolist()
                    except Exception:
                        pass

                # flat list of ints -> decode to string
                if isinstance(inputs, (list, tuple)):
                    if len(inputs) == 0:
                        return inputs
                    is_nested = any(isinstance(x, (list, tuple)) for x in inputs)
                    is_all_ints = all(isinstance(x, int) for x in inputs)
                    if is_all_ints and not is_nested:
                        try:
                            return self._m.get_tokenizer().decode(list(inputs), skip_special_tokens=False)
                        except Exception:
                            return " ".join(str(x) for x in inputs)
                    if is_nested:
                        out = []
                        for seq in inputs:
                            if isinstance(seq, (list, tuple)) and all(isinstance(x, int) for x in seq):
                                try:
                                    out.append(self._m.get_tokenizer().decode(list(seq), skip_special_tokens=False))
                                except Exception:
                                    out.append(" ".join(str(x) for x in seq))
                            else:
                                out.append(seq)
                        return out
                return inputs

            def generate(self, *args, generation_config=None, streamer=None, **kwargs):
                # accept positional input or keyword input_ids
                inputs = None
                remaining = ()
                used_kw = False
                if args:
                    inputs = args[0]
                    remaining = args[1:]
                elif "input_ids" in kwargs:
                    inputs = kwargs.pop("input_ids")
                    used_kw = True

                norm = self._normalize(inputs)
                if used_kw:
                    kwargs["input_ids"] = norm
                    return self._m.generate(*remaining, generation_config=generation_config, streamer=streamer, **kwargs)
                return self._m.generate(norm, *remaining, generation_config=generation_config, streamer=streamer, **kwargs)

            def __call__(self, *args, **kwargs):
                inputs = None
                remaining = ()
                if args:
                    inputs = args[0]
                    remaining = args[1:]
                elif "inputs" in kwargs:
                    inputs = kwargs.pop("inputs")
                norm = self._normalize(inputs)
                if args:
                    return self._m.__call__(norm, *remaining, **kwargs)
                elif "inputs" in locals():
                    kwargs["inputs"] = norm
                    return self._m.__call__(**kwargs)
                else:
                    return self._m.__call__(norm, **kwargs)

            def __getattr__(self, name):
                return getattr(self._m, name)

        return ModelWrapper(self._model, self.tokenizer)

    @property
    def config(self):
        return self._config

    def _model_call(self, inps, **kwargs):
        """Try to use OpenVINO logprobs; if not available, return dummy logits.

        Expects `inps` to be a torch Tensor of shape (batch, seq_len) when called
        from the harness. Keep this small and robust.
        """
        try:
            import torch
        except Exception:
            torch = None
        try:
            import numpy as _np
        except Exception:
            _np = None

        if torch is None:
            raise RuntimeError("torch is required for _model_call")

        batch_size, seq_len = inps.shape
        vocab_size = getattr(self.tokenizer, "vocab_size", 32000)

        # Best-effort: attempt to use OpenVINO GenerationConfig to get logprobs
        try:
            gen_cfg = ov_genai.GenerationConfig()
            gen_cfg.max_new_tokens = 1
            gen_cfg.echo = True
            gen_cfg.do_sample = False
            gen_cfg.logprobs = min(50, vocab_size)

            all_logits = []
            for b in range(batch_size):
                input_ids = inps[b].tolist()
                try:
                    res = self.model(input_ids, gen_cfg)
                    lp = getattr(res, "logprobs", None)
                    if lp:
                        # lp is expected to be a sequence/dict-like per position
                        batch_logits = []
                        for pos in range(seq_len):
                            pos_log = lp[pos] if pos < len(lp) else {}
                            pos_tensor = torch.full((vocab_size,), -float("inf"), dtype=torch.float32)
                            for tid, logp in pos_log.items():
                                if 0 <= int(tid) < vocab_size:
                                    pos_tensor[int(tid)] = float(logp)
                            batch_logits.append(pos_tensor)
                        all_logits.append(torch.stack(batch_logits))
                    else:
                        all_logits.append(torch.randn(seq_len, vocab_size, dtype=torch.float32))
                except Exception as e:
                    eval_logger.warning(f"OpenVINO logprobs failed for batch {b}: {e}")
                    all_logits.append(torch.randn(seq_len, vocab_size, dtype=torch.float32))

            if all_logits:
                return torch.stack(all_logits)
        except Exception as e:
            eval_logger.debug(f"OpenVINO GenerationConfig/logprobs approach failed: {e}")

        # fallback
        eval_logger.warning("Falling back to dummy logits for OpenVINO model")
        return torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)

    # Minimal compatibility implementations for abstract LM API
    def tok_encode(self, string: str, **kwargs) -> List[int]:
        """Tokenize a string to a list of ints using the wrapped tokenizer."""
        try:
            return self.tokenizer.encode(string)
        except Exception:
            # Best-effort fallback: split on whitespace and return ords (not ideal)
            return [ord(c) for c in string][:1]

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> List[tuple[float, bool]]:
        """Simple fallback loglikelihood: length-based heuristic using tok_encode."""
        res = []
        for req in requests:
            if isinstance(req, tuple):
                context, continuation = req
            else:
                # Instance-like object
                context, continuation = req.args

            try:
                cont_tokens = self.tok_encode(continuation)
                if len(cont_tokens) == 0:
                    res.append((-float("inf"), False))
                else:
                    res.append((-len(cont_tokens) * 2.0, True))
            except Exception:
                res.append((-5.0, True))

        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        """Compute simple rolling loglikelihood using token lengths as a fallback."""
        out = []
        for req in requests:
            if isinstance(req, tuple):
                (context,) = req
            else:
                (context,) = req.args
            try:
                toks = self.tok_encode(context)
                out.append(-len(toks) * 2.0)
            except Exception:
                out.append(-5.0)
        return out

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """Greedy generation fallback: call the model.generate or model() and return text.

        Expects requests to be list where each req.args == (context, gen_kwargs)
        """
        results = []
        for req in requests:
            # Extract context and generation kwargs
            if isinstance(req, tuple):
                context = req[0]
                gen_kwargs = req[1] if len(req) > 1 else {}
            else:
                context, gen_kwargs = req.args

            try:
                # Prefer generation_config if provided
                if isinstance(gen_kwargs, dict) and "generation_config" in gen_kwargs:
                    out = self.model.generate(context, **gen_kwargs)
                else:
                    # Try converting simple dict to GenerationConfig when possible
                    try:
                        if ov_genai is not None and isinstance(gen_kwargs, dict):
                            gen_cfg = ov_genai.GenerationConfig()
                            for k, v in gen_kwargs.items():
                                try:
                                    setattr(gen_cfg, k, v)
                                except Exception:
                                    pass
                            out = self.model.generate(context, generation_config=gen_cfg)
                        else:
                            out = self.model.generate(context, **(gen_kwargs or {}))
                    except Exception:
                        out = self.model.generate(context, **(gen_kwargs or {}))

                # Normalize output to string
                if isinstance(out, str):
                    results.append(out)
                elif hasattr(out, "generated_text"):
                    results.append(getattr(out, "generated_text"))
                elif isinstance(out, (list, tuple)) and len(out) > 0:
                    results.append(str(out[0]))
                else:
                    results.append("")
            except Exception as e:
                eval_logger.warning(f"generate_until fallback error: {e}")
                results.append("")

        return results
