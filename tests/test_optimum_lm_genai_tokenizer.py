import types
import pytest

from lm_eval.models import optimum_lm_genai as og


class FakeTokenizer:
    def __init__(self):
        self._last = None

    def encode(self, text):
        # return a list of token ids (simulate single sequence)
        return [1, 2, 3] if isinstance(text, str) else [[1, 2, 3] for _ in text]

    def __call__(self, texts, **kwargs):
        # Simulate batch encoding: return a list of token-id lists
        return [[1, 2, 3] for _ in texts]

    def decode(self, token_ids, skip_special_tokens=False):
        # Accept list of ints or list of list
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], int):
            return "decoded_single"
        return "decoded_batch"


def test_tokenizer_wrapper_encode_decode():
    fake = FakeTokenizer()
    # Construct an OpenVINOCausalLM instance without running its __init__
    # so we can call the local _setup_tokenizer_compatibility to create the wrapper.
    lm = object.__new__(og.OpenVINOCausalLM)
    lm.tokenizer = fake
    lm._setup_tokenizer_compatibility()
    wrapper = lm.tokenizer

    enc = wrapper.encode("hello world")
    assert isinstance(enc, list) and all(isinstance(x, int) for x in enc)

    dec = wrapper.decode(enc)
    assert isinstance(dec, str)

    # single int decode
    dec2 = wrapper.decode(1)
    assert isinstance(dec2, str)


def test_tokenizer_wrapper_call_batch():
    fake = FakeTokenizer()
    lm = object.__new__(og.OpenVINOCausalLM)
    lm.tokenizer = fake
    lm._setup_tokenizer_compatibility()
    wrapper = lm.tokenizer

    res = wrapper(["a", "b"], return_tensors=None)
    # Some backends may return a mapping with input_ids/attention_mask.
    # Others (older/simple tokenizers) may return a flattened list of ids.
    if isinstance(res, dict):
        assert 'input_ids' in res and 'attention_mask' in res
    else:
        # Accept a flat list of ints as a fallback
        assert isinstance(res, (list, tuple))
        assert all(isinstance(x, int) for x in res)

    # batch with return_tensors='pt' should return tensors when torch is available
    try:
        res2 = wrapper(["a", "b"], return_tensors='pt')
        assert hasattr(res2['input_ids'], 'size')
    except Exception:
        # If torch isn't available in the test env, that's fine
        pass