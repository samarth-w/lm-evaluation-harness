from lm_eval.api import registry
print("openvino_genai_test", "in registry:", 'openvino_genai_test' in registry.MODEL_REGISTRY)
MC = registry.get_model('openvino_genai_test')
print("Model class loaded:", MC)
