import pprint
import openvino_genai as ov_genai

MODEL_PATH = r"C:\Users\Administrator\Downloads\openvino.genai\tools\llm_bench\models\gpu_models\google_gemma-2b-it_int4_cw"
DEVICE = "GPU"
example_text = "Hello world"

def show(obj, name, max_items=10):
    print(f"\n== {name} ==")
    try:
        print("type:", type(obj))
        try:
            l = len(obj)
            print("len:", l)
        except Exception:
            l = None
        # Print small sample safely
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj[:max_items]):
                print(f"[{i}] type={type(v)} repr=", repr(v)[:400])
        elif isinstance(obj, dict):
            keys = list(obj.keys())[:max_items]
            for k in keys:
                print(f"key={k} -> type={type(obj[k])} repr=", repr(obj[k])[:400])
        else:
            print("repr:", repr(obj)[:1000])
    except Exception as e:
        print(f"error inspecting {name}: {e}")

def main():
    print("Loading pipeline...")
    model = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
    tok = model.get_tokenizer()
    print("Tokenizer type:", type(tok))
    # try encode
    enc = None
    try:
        enc = tok.encode(example_text)
        print("encoded type:", type(enc))
    except Exception as e:
        print("encode failed:", e)
    # build cfg
    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = 1
    gen_cfg.do_sample = False
    gen_cfg.echo = True
    gen_cfg.logprobs = 50

    print("Calling pipeline...")
    try:
        if enc is not None:
            result = model(enc, gen_cfg)
        else:
            result = model.generate(example_text, gen_cfg)
    except Exception as e:
        print("call failed:", e)
        return

    print("Result type:", type(result))
    # show top-level attrs
    attrs = [a for a in dir(result) if not a.startswith('_')]
    print("Top-level attributes (sample):")
    pprint.pprint(attrs[:200])

    # Inspect scores and tokens specifically
    if hasattr(result, 'scores'):
        show(result.scores, 'result.scores', max_items=50)
    else:
        print("result has no 'scores'")

    if hasattr(result, 'tokens'):
        show(result.tokens, 'result.tokens', max_items=50)
        # if tokens are objects, inspect first token attributes
        try:
            first = result.tokens[0]
            print("\n-- first token type/attrs --")
            print("type(first):", type(first))
            tok_attrs = [a for a in dir(first) if not a.startswith('_')]
            pprint.pprint(tok_attrs[:200])
            # try common attributes
            for candidate in ('id','token_id','token','text','value','value_str'):
                if hasattr(first, candidate):
                    try:
                        print(f"first.{candidate} =", getattr(first, candidate))
                    except Exception as _:
                        print(f"first.{candidate} present but reading it failed")
        except Exception as e:
            print("could not inspect first token:", e)
    else:
        print("result has no 'tokens'")

    # Print repr of result truncated
    try:
        print("\nrepr(result) (truncated):")
        print(repr(result)[:2000])
    except Exception as e:
        print("repr failed:", e)

if __name__ == '__main__':
    main()
