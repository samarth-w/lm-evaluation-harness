import os
import pytest


try:
    import openvino_genai  # type: ignore
    _HAS_OV = True
except Exception:
    _HAS_OV = False


@pytest.mark.skipif(not _HAS_OV, reason="OpenVINO GenAI not installed")
def test_optimum_genai_loglikelihood_real():
    """Smoke test: ensure loglikelihood returns a numeric, non-placeholder logprob.

    This test requires a working OpenVINO GenAI installation and a pretrained model
    available via the environment variable OPENVINO_PRETRAINED_MODEL. The test will
    be skipped if these are not present so it can run safely in CI where OpenVINO
    is not available.
    """
    pretrained = os.getenv("OPENVINO_PRETRAINED_MODEL")
    if not pretrained:
        pytest.skip("Set OPENVINO_PRETRAINED_MODEL to run OpenVINO GenAI tests")

    from lm_eval.models.optimum_lm_genai import OpenVINOCausalLM
    from lm_eval.api.instance import Instance

    # Create model (may download/cache model as needed)
    model = OpenVINOCausalLM(pretrained=pretrained, device=os.getenv("OPENVINO_DEVICE", "cpu"))

    # Small, trivial request
    context = "The sky is"
    continuation = " blue."
    inst = Instance(request_type="loglikelihood", doc={}, arguments=(context, continuation), idx=0)

    # Call loglikelihood (disable tqdm to keep test output clean)
    res = model.loglikelihood([inst], disable_tqdm=True)

    assert isinstance(res, list) and len(res) == 1
    logprob, is_greedy = res[0]

    # Sanity checks: numeric, not the error placeholder used in the adapter, and plausible
    assert isinstance(logprob, float)
    assert logprob != -100.0
    # Log-probabilities for normal continuations are typically negative
    assert logprob < 0.0
    assert isinstance(is_greedy, bool)
