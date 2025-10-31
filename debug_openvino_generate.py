import time
import openvino_genai as ov

MODEL = r"C:\Users\Administrator\Downloads\openvino.genai\tools\llm_bench\models\gpu_models\google_gemma-2b-it_int4_cw"

pipe = ov.LLMPipeline(MODEL, "GPU")
tok = pipe.get_tokenizer()
prompt = "Translate to Python: print(1+1)\n"  # small prompt for a fast test
print("Tokenizer type:", type(tok))
print("Decoding short tokenized test...")

# Try pipeline.generate with short max_length
start = time.time()
try:
    res = pipe.generate(prompt, max_length=16, do_sample=False)
    print("generate returned in", time.time() - start, "s")
    print("Result:", res)
except Exception as e:
    print("generate raised:", repr(e))
