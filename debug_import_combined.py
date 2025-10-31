import importlib, importlib.util, traceback, sys
from lm_eval import api

modpath = r"c:\Users\Administrator\Downloads\lm-evaluation-harness\lm_eval\models\optimum_lm_genai_combined.py"
print("Loading module from:", modpath)

try:
    spec = importlib.util.spec_from_file_location("lm_eval.models.optimum_lm_genai_combined", modpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("Module imported OK")
except Exception:
    print("Module import FAILED")
    traceback.print_exc(file=sys.stdout)

from lm_eval.api import registry
print("Registered present keys (first 40):", list(registry.MODEL_REGISTRY.keys())[:40])
print("'openvino_genai_test' registered?:", 'openvino_genai_test' in registry.MODEL_REGISTRY)
