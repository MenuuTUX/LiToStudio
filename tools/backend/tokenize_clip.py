#!/usr/bin/env python3
"""CLIP-BPE tokenizer worker — turns free-text concept phrases into the token-ID
arrays the SAM 3.1 CoreML text encoder expects (the native Swift driver has no
tokenizer, and the 12 taxonomy prompts are pre-baked; this covers user free-text).

argv[1] = manifest JSON: {"phrases": ["chain belt", "red ribbon"]}
stdout  = {"ok": true, "tokenizer": "...", "tokens": [[49406, ...32 ids...], ...]}
"""
import sys, json, traceback

def main():
    manifest = json.load(open(sys.argv[1]))
    phrases = manifest["phrases"]
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens = [tok(p, padding="max_length", max_length=32, truncation=True)["input_ids"]
              for p in phrases]
    print(json.dumps({"ok": True,
                      "tokenizer": "openai/clip-vit-base-patch32 (CLIP BPE, max_length 32)",
                      "tokens": tokens}))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e),
                          "instruction": "ensure tools/backend/.venv exists (setup.sh)"}))
