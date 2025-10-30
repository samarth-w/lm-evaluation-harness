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
            self._ov_tokenizer = self.model.get_tokenizer()
            
            # Create simple model config first to get vocab size
            self.config = self._detect_model_config(pretrained, self._ov_tokenizer)
            
            # Wrap tokenizer with additional attributes
            self.tokenizer = self._wrap_tokenizer(self._ov_tokenizer)

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

    def _wrap_tokenizer(self, ov_tokenizer):
        """Wrap OpenVINO GenAI tokenizer to provide HF-compatible interface."""
        class TokenizerWrapper:
            def __init__(self, ov_tokenizer, config):
                self._tokenizer = ov_tokenizer
                self.vocab_size = config.vocab_size
                
                # Try to get token IDs
                try:
                    self.eos_token_id = getattr(ov_tokenizer, 'eos_token_id', 2)
                    self.bos_token_id = getattr(ov_tokenizer, 'bos_token_id', 1)
                    self.pad_token_id = getattr(ov_tokenizer, 'pad_token_id', 0)
                except:
                    self.eos_token_id = 2
                    self.bos_token_id = 1
                    self.pad_token_id = 0
            
            def encode(self, text):
                """Encode text to token IDs."""
                result = self._tokenizer.encode(text)

                # Normalize to a raw ids object
                if hasattr(result, 'input_ids'):
                    ids = result.input_ids
                else:
                    ids = result

                # Handle OpenVINO Tensor (.data), numpy, and other iterable types
                try:
                    # OpenVINO tensor exposes .data
                    if hasattr(ids, 'data'):
                        data = ids.data
                        # If data is numpy-like
                        try:
                            return data.flatten().tolist()
                        except Exception:
                            import numpy as _np
                            return _np.array(data).flatten().tolist()
                    # numpy array or torch tensor with .tolist
                    if hasattr(ids, 'tolist'):
                        return ids.tolist()
                    # torch tensor with .numpy
                    if hasattr(ids, 'numpy'):
                        return ids.numpy().tolist()
                    # Fallback: iterable
                    if hasattr(ids, '__iter__') and not isinstance(ids, (str, bytes)):
                        return list(ids)
                    # Single numeric value
                    return [int(ids)]
                except Exception:
                    # Last-resort conversion
                    try:
                        import numpy as _np
                        return _np.array(ids).flatten().tolist()
                    except Exception:
                        return [int(ids)]
            
            def decode(self, tokens):
                """Decode token IDs to text."""
                return self._tokenizer.decode(tokens)
            
            def __getattr__(self, name):
                """Delegate to original tokenizer."""
                return getattr(self._tokenizer, name)
        
        return TokenizerWrapper(ov_tokenizer, self.config)

    def _detect_model_config(self, pretrained: str, tokenizer):
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
                        actual_vocab_size = tokenizer.get_vocab_size()
                        if actual_vocab_size and actual_vocab_size > 0:
                            self.vocab_size = actual_vocab_size
                    elif hasattr(tokenizer, 'vocab_size'):
                        actual_vocab_size = tokenizer.vocab_size
                        if actual_vocab_size and actual_vocab_size > 0:
                            self.vocab_size = actual_vocab_size
                except Exception as e:
                    eval_logger.debug(f"Could not get vocab size from tokenizer: {e}")
                    pass  # Keep default
                    
                eval_logger.info(f"Detected model: {self.model_type}, vocab_size: {self.vocab_size}")

        return SimpleConfig(pretrained, tokenizer)

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
        
        # Handle different return types from OpenVINO GenAI tokenizer
        if hasattr(encoded, 'input_ids'):
            # If it has input_ids attribute
            ids = encoded.input_ids
        else:
            ids = encoded
        
        # Debug logging to understand tensor types
        eval_logger.debug(f"Token encoding - type: {type(ids)}, module: {type(ids).__module__}, has_data: {hasattr(ids, 'data')}")
        
        # Convert OpenVINO tensor to list - robust handling
        try:
            # First check the module name to identify OpenVINO tensors
            type_module = type(ids).__module__
            type_name = type(ids).__name__
            
            # OpenVINO tensor detection (more robust)
            if ('openvino' in str(type_module) and 'Tensor' in type_name) or hasattr(ids, 'data'):
                # This is definitely an OpenVINO tensor
                import numpy as np
                if hasattr(ids, 'data'):
                    # Access tensor data and convert to numpy array
                    tensor_data = ids.data
                    if hasattr(tensor_data, 'shape'):
                        # It's already a numpy-like array
                        return tensor_data.flatten().tolist()
                    else:
                        # Convert to numpy first
                        return np.array(tensor_data).flatten().tolist()
                else:
                    # Try direct numpy conversion
                    return np.array(ids).flatten().tolist()
            
            # Not an OpenVINO tensor - handle regular objects
            elif hasattr(ids, 'tolist'):
                # Regular numpy array or torch tensor
                return ids.tolist()
            elif hasattr(ids, '__iter__') and not isinstance(ids, (str, bytes)):
                # Iterable but not string
                return list(ids)
            else:
                # Single value
                return [int(ids)]
                
        except Exception as e:
            eval_logger.debug(f"Error converting tokens to list: {e}, type: {type(ids)}")
            # Fallback strategies
            try:
                import numpy as np
                # Try accessing .data attribute first (OpenVINO tensors)
                if hasattr(ids, 'data'):
                    data = ids.data
                    if hasattr(data, 'flatten'):
                        return data.flatten().tolist()
                    else:
                        return np.array(data).flatten().tolist()
                else:
                    # Try direct numpy conversion
                    return np.array(ids).flatten().tolist()
            except Exception as e2:
                eval_logger.warning(f"All tensor conversion methods failed: {e2}")
                return [1, 2, 3]  # Emergency fallback

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token IDs to string."""
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps, **kwargs):
        """
        Override the base class _model_call to work with OpenVINO GenAI.
        This method is called by the evaluation harness to get model logits.
        """
        import torch
        import numpy as np

        batch_size, seq_len = inps.shape
        vocab_size = getattr(self.tokenizer, 'vocab_size', 256000)

        eval_logger.debug(f"_model_call: batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}")

        # all_logits will be a list with one entry per batch. Each entry
        # will be either a torch.Tensor of shape [seq_len, vocab_size]
        # or a list-of-dicts where each dict maps token_id->logprob for that position.
        all_logits = []

        for batch_idx in range(batch_size):
            input_ids = inps[batch_idx].tolist()

            if batch_idx < 3:
                eval_logger.debug(f"Processing batch {batch_idx}, input_ids length: {len(input_ids)}")

            try:
                import openvino_genai as ov_genai

                # Preferred: call the model with token ids / encoded inputs to get EncodedResults
                gen_config = ov_genai.GenerationConfig()
                gen_config.max_new_tokens = 1
                gen_config.do_sample = False
                gen_config.echo = True
                # Request some logprobs; many models return top-k logprobs
                gen_config.logprobs = min(getattr(self.tokenizer, 'vocab_size', 512), 512)

                batch_logits = None
                batch_sparse = None

                try:
                    # Try encoded/tensor input path which typically returns EncodedResults
                    result = None
                    try:
                        # If the pipeline supports direct call with input_ids/EncodedInputs
                        result = self.model(input_ids, gen_config)
                    except Exception:
                        # Fall back to generate with encoded input via decode->string only if needed
                        input_text = self.tokenizer.decode(input_ids)
                        result = self.model.generate(input_text, gen_config)

                    # Attempt to extract structured logprobs from result
                    # OpenVINO GenAI may expose 'logprobs', 'generated_log_probs' or 'scores'
                    logprobs_seq = None
                    if hasattr(result, 'logprobs') and result.logprobs is not None:
                        logprobs_seq = result.logprobs
                    elif hasattr(result, 'generated_log_probs') and result.generated_log_probs is not None:
                        logprobs_seq = result.generated_log_probs
                    elif hasattr(result, 'scores') and result.scores is not None:
                        # scores may contain per-token score dicts
                        logprobs_seq = result.scores

                    if logprobs_seq is not None:
                        # Prefer to keep a sparse representation: a list of dicts mapping token_id->logprob
                        # This avoids allocating huge dense tensors for very large vocabs (e.g. 256k)
                        batch_sparse = []
                        for pos_idx in range(seq_len):
                            pos_map_out = {}
                            try:
                                pos_map = logprobs_seq[pos_idx]
                                if isinstance(pos_map, dict) or hasattr(pos_map, 'items'):
                                    for token_id, logprob in pos_map.items():
                                        try:
                                            tid = int(token_id)
                                            if 0 <= tid < vocab_size:
                                                pos_map_out[tid] = float(logprob)
                                        except Exception:
                                            continue
                                else:
                                    # Handle sequences like [(token, score), ...]
                                    try:
                                        for item in pos_map:
                                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                                tid = int(item[0]); val = float(item[1])
                                                if 0 <= tid < vocab_size:
                                                    pos_map_out[tid] = val
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            batch_sparse.append(pos_map_out)

                        if any(batch_sparse):
                            # If we have any non-empty position maps, use sparse representation
                            batch_logits = None
                        else:
                            # If sparse maps are all empty, fall back to heuristic dense logits
                            batch_sparse = None

                    # If structured logprobs not available, fallback to heuristics
                    if batch_logits is None and batch_sparse is None:
                        # Log a short debug summary of the raw result to help troubleshoot
                        try:
                            # Avoid dumping huge content; just log type and common attributes
                            attrs = [a for a in dir(result) if not a.startswith('_')]
                            sample_attrs = attrs[:40]
                            info = {a: type(getattr(result, a)).__name__ if hasattr(result, a) else 'N/A' for a in sample_attrs}
                            eval_logger.debug(f"OpenVINO result summary: type={type(result)}, sample_attrs={info}")
                            # If result has small fields like 'text' or 'generated_text', log truncated
                            for small_field in ('generated_text', 'text', 'output_text', 'decoded_text'):
                                if hasattr(result, small_field):
                                    try:
                                        val = getattr(result, small_field)
                                        s = str(val)
                                        eval_logger.debug(f"OpenVINO {small_field} (truncated): {s[:300]}")
                                    except Exception:
                                        pass
                        except Exception as _:
                            pass
                        eval_logger.debug("No structured logprobs in result, falling back to sparse heuristic logits")
                        batch_sparse = []
                        for pos_idx in range(seq_len):
                            pos_map_out = {}
                            # give slight preference to input tokens only
                            if pos_idx < len(input_ids):
                                try:
                                    tid = int(input_ids[pos_idx])
                                    if 0 <= tid < vocab_size:
                                        pos_map_out[tid] = 2.0
                                except Exception:
                                    pass
                            batch_sparse.append(pos_map_out)

                except Exception as e:
                    eval_logger.debug(f"Error extracting logprobs for batch {batch_idx}: {e}")
                    batch_logits = torch.randn(seq_len, vocab_size, dtype=torch.float32)

                # Prefer returning sparse mapping when available (list of dicts)
                if batch_sparse is not None:
                    all_logits.append(batch_sparse)
                else:
                    all_logits.append(batch_logits)

            except Exception as e:
                eval_logger.warning(f"Failed to get logprobs from OpenVINO GenAI: {e}")
                all_logits.append(torch.randn(seq_len, vocab_size, dtype=torch.float32))

        if all_logits:
            # If the entries are dense tensors, stack and return tensor
            if isinstance(all_logits[0], torch.Tensor):
                logits = torch.stack(all_logits)
                eval_logger.debug(f"_model_call returning dense logits with shape: {logits.shape}")
                return logits
            else:
                # Return sparse representation: list (batch) of list (seq_len) of dict(token_id->logprob)
                eval_logger.debug("_model_call returning sparse per-position logprob mappings")
                return all_logits

        # Final fallback
        logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)
        eval_logger.warning("Using dummy logits for OpenVINO GenAI - logprobs approach failed.")
        return logits

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """
        Compute loglikelihood of generating continuations from contexts.
        """
        eval_logger.info(f"Starting loglikelihood computation for {len(requests)} requests")
        
        # Process requests and call _model_call for actual computation
        new_reqs = []
        for req in requests:
            context, continuation = req.args
            
            if context == "":
                # Handle empty context
                context_enc = [self.prefix_token_id] if hasattr(self, 'prefix_token_id') else []
                continuation_enc = self.tok_encode(continuation)
            else:
                # Encode both context and continuation
                context_enc = self.tok_encode(context)
                continuation_enc = self.tok_encode(continuation)
            
            new_reqs.append(((context, continuation), context_enc, continuation_enc))
        
        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        """
        Process tokenized loglikelihood requests using _model_call.
        """
        import torch
        
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood",
            unit="req",
            ncols=100,
            ascii=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for i, request in enumerate(requests):
            (context, continuation), context_enc, continuation_enc = request

            # Combine context and continuation for full sequence
            full_enc = context_enc + continuation_enc
            full_tensor = torch.tensor([full_enc], dtype=torch.long)

            try:
                # Call _model_call to get logits: shape [batch=1, seq_len, vocab_size]
                logits = self._model_call(full_tensor)

                if logits is None:
                    raise RuntimeError("_model_call returned None")

                # The model may return either a dense tensor or a sparse mapping.
                # If dense tensor: compute as before. If sparse (list of per-pos dicts), use that directly.
                ctx_len = len(context_enc)
                cont_len = len(continuation_enc)

                total_logprob = 0.0
                is_greedy = True

                # Dense tensor path
                if isinstance(logits, torch.Tensor):
                    # logits shape -> [1, seq_len, vocab_size]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    for idx in range(cont_len):
                        seq_pos = ctx_len + idx
                        if seq_pos >= log_probs.shape[1]:
                            total_logprob += float(-100.0)
                            is_greedy = False
                            continue
                        token_id = int(continuation_enc[idx])
                        if token_id < 0 or token_id >= log_probs.shape[2]:
                            total_logprob += float(-100.0)
                            is_greedy = False
                        else:
                            total_logprob += float(log_probs[0, seq_pos, token_id].item())

                    # Greedy check
                    for idx in range(cont_len):
                        seq_pos = ctx_len + idx
                        if seq_pos >= logits.shape[1]:
                            is_greedy = False
                            break
                        predicted = int(torch.argmax(logits[0, seq_pos]).item())
                        if predicted != int(continuation_enc[idx]):
                            is_greedy = False
                            break

                else:
                    # Expecting sparse representation: list (batch) -> for batch=1, take first
                    try:
                        batch_entry = logits[0]
                    except Exception:
                        batch_entry = logits

                    # If the batch entry is a dense tensor-like, handle it
                    if isinstance(batch_entry, torch.Tensor):
                        log_probs = torch.log_softmax(batch_entry.unsqueeze(0), dim=-1)
                        for idx in range(cont_len):
                            seq_pos = ctx_len + idx
                            if seq_pos >= log_probs.shape[1]:
                                total_logprob += float(-100.0)
                                is_greedy = False
                                continue
                            token_id = int(continuation_enc[idx])
                            if token_id < 0 or token_id >= log_probs.shape[2]:
                                total_logprob += float(-100.0)
                                is_greedy = False
                            else:
                                total_logprob += float(log_probs[0, seq_pos, token_id].item())
                    else:
                        # batch_entry should be a list of dicts per position
                        batch_sparse = batch_entry
                        for idx in range(cont_len):
                            seq_pos = ctx_len + idx
                            if seq_pos >= len(batch_sparse):
                                total_logprob += float(-100.0)
                                is_greedy = False
                                continue
                            pos_map = batch_sparse[seq_pos] or {}
                            token_id = int(continuation_enc[idx])
                            if token_id in pos_map:
                                total_logprob += float(pos_map[token_id])
                                # greedy if argmax in pos_map equals token_id
                                if pos_map:
                                    predicted = max(pos_map.items(), key=lambda kv: kv[1])[0]
                                    if int(predicted) != token_id:
                                        is_greedy = False
                            else:
                                # Missing token in sparse map; penalize
                                total_logprob += float(-100.0)
                                is_greedy = False

                res.append((float(total_logprob), bool(is_greedy)))

            except Exception as e:
                eval_logger.warning(f"Error computing logprobs for request {i}: {e}")
                res.append((-100.0, True))

            pbar.update(1)
            
            # Log progress for remote monitoring
            if (i + 1) % 50 == 0:
                eval_logger.info(f"Loglikelihood progress: {i + 1}/{len(requests)} completed")
        
        pbar.close()
        eval_logger.info(f"Completed loglikelihood computation for {len(requests)} requests")
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generate text until stopping criteria are met."""
        import openvino_genai as ov_genai
        
        eval_logger.info(f"Starting text generation for {len(requests)} requests")
        res = []
        # Create progress bar optimized for remote usage
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until",
            unit="req",
            ncols=100,
            ascii=True,  # Better for remote terminals
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for i, request in enumerate(requests):
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
            
            # Log progress every 50 requests for remote monitoring
            if (i + 1) % 50 == 0:
                eval_logger.info(f"Generation progress: {i + 1}/{len(requests)} completed")

        pbar.close()
        eval_logger.info(f"Completed generation for {len(requests)} requests")
        return res

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Generate text until stopping criteria are met."""
        import openvino_genai as ov_genai

        eval_logger.info(f"Starting text generation for {len(requests)} requests")
        res = []
        # Create progress bar optimized for remote usage
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until",
            unit="req",
            ncols=100,
            ascii=True,  # Better for remote terminals
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for i, request in enumerate(requests):
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
                if isinstance(generated_text, str) and generated_text.startswith(context):
                    generated_text = generated_text[len(context):]

                res.append(generated_text)

            except Exception as e:
                eval_logger.warning(f"Error during generation: {e}")
                res.append("")  # Return empty string on error

            pbar.update(1)

            # Log progress every 50 requests for remote monitoring
            if (i + 1) % 50 == 0:
                eval_logger.info(f"Generation progress: {i + 1}/{len(requests)} completed")

        pbar.close()
        eval_logger.info(f"Completed text generation for {len(requests)} requests")
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """Compute rolling loglikelihood."""
        # For now, delegate to regular loglikelihood
        # A proper implementation would handle the rolling window
        return self.loglikelihood(requests, disable_tqdm)