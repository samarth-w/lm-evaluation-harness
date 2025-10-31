import pprint
import openvino_genai as ov_genai

MODEL_PATH = r"C:\Users\Administrator\Downloads\openvino.genai\tools\llm_bench\models\gpu_models\google_gemma-2b-it_int4_cw"
DEVICE = "GPU"

def main():
    print("Loading pipeline...")
    model = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
    tok = model.get_tokenizer()
    print("Tokenizer type:", type(tok))
    example_text = "Hello world"

    # Try encoding in the same way the harness does
    try:
        enc = tok.encode(example_text)
        print("encoded type:", type(enc))
        try:
            # attempt to show first tokens if iterable
            print("sample encoded (first 40 tokens):", list(enc)[:40])
        except Exception:
            print("encoded repr (truncated):", repr(enc)[:500])
    except Exception as e:
        print("tokenizer.encode() failed:", e)
        enc = None

    # Build generation config requesting logprobs
    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = 1
    gen_cfg.do_sample = False
    gen_cfg.echo = True
    gen_cfg.logprobs = 50  # ask for top-k logprobs

    print("Calling pipeline with encoded input (if possible)...")
    result = None
    try:
        if enc is not None:
            # try raw encoded path
            result = model(enc, gen_cfg)
        else:
            raise RuntimeError("No encoded input available")
    except Exception as e:
        print("Encoded call failed:", e)
        print("Falling back to string generate(...) call")
        try:
            result = model.generate(example_text, gen_cfg)
        except Exception as e2:
            print("String generate failed:", e2)
            raise

    print("Result type:", type(result))
    try:
        attrs = [a for a in dir(result) if not a.startswith('_')]
        print("Top-level attributes (sample):")
        pprint.pprint(attrs[:80])
    except Exception as e:
        print("dir(result) failed:", e)

    # Check likely fields
    for name in ("logprobs", "generated_log_probs", "scores", "outputs", "generations", "text", "generated_text"):
        if hasattr(result, name):
            val = getattr(result, name)
            print(f"\\nField '{name}' present, type:", type(val))
            try:
                if isinstance(val, (list, dict)):
                    # print a small sample safely
                    if isinstance(val, list):
                        pprint.pprint(val[:5])
                    else:
                        sample_keys = list(val)[:10]
                        pprint.pprint({k: val[k] for k in sample_keys})
                else:
                    s = str(val)
                    print("repr (truncated):", s[:1000])
            except Exception as e:
                print(f"Error printing '{name}':", e)

    # Also print a short repr of result to capture any nested types
    try:
        print("\\nrepr(result) (truncated):")
        r = repr(result)
        print(r[:2000])
    except Exception as e:
        print("repr(result) failed:", e)

if __name__ == "__main__":
    main()
