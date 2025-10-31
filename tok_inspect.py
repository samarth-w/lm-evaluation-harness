import openvino_genai as ov, inspect, traceback
pretrained = r"C:\Users\Administrator\Downloads\openvino.genai\tools\llm_bench\models\gpu_models\google_gemma-2b-it_int4_cw"
print("Loading LLMPipeline (this may take a little while)...")
pipe = ov.LLMPipeline(pretrained, "GPU")
tok = pipe.get_tokenizer()
print("TOKENIZER TYPE:", type(tok))
print("\nSAMPLE public attributes:")
print([n for n in dir(tok) if not n.startswith('_')][:200])
print("\nHas padding_side?:", hasattr(tok, 'padding_side'))
print('Has encode?:', hasattr(tok, 'encode'), ' Has decode?:', hasattr(tok, 'decode'))
print('\n--- Attempt tok.decode(1) ---')
try:
    out = tok.decode(1)
    print('decode(1) ->', out)
except Exception:
    traceback.print_exc()
print('\n--- Attempt tok.decode([1]) ---')
try:
    out = tok.decode([1])
    print('decode([1]) ->', out)
except Exception:
    traceback.print_exc()
