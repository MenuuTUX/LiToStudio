#!/usr/bin/env python3
"""Pre-tokenize the taxonomy core-set prompts for the SAM3.1 CoreML text encoder.

The CoreML TextEncoder takes token_ids (1, 32) — no tokenizer ships with the
conversion. SAM3's text tower follows the CLIP BPE convention (49406 <start>,
49407 <end>, pad 0... CLIP pads with the EOT token id in HF's tokenizer when
padding='max_length' — we record exactly what HF CLIPTokenizer emits and validate
the whole chain empirically against real images (LiToSmoke sam3). The prompt set is
fixed (taxonomy core set), so baking IDs avoids needing a Swift tokenizer.

Output: weights/sam3-coreml/prompt_tokens.json
"""
import json, os, sys

PROMPTS = [
    ("L001", "face_region", "face"),
    ("L002", "hair_volume", "hair"),
    ("L003", "upper_torso_garment", "swimsuit top"),
    ("L004", "abdomen_navel_region", "abdomen"),
    ("L005", "navel_piercing", "navel piercing"),
    ("L006", "chain_belt", "chain belt"),
    ("L007", "hip_strap_line", "hip strap"),
    ("L008", "gloves", "fingerless gloves"),
    ("L009", "cargo_pants", "cargo pants"),
    ("L010", "pants_pockets", "pants pocket"),
    ("L011", "belt_charms", "charm pendant"),
    ("L012", "fingernails", "fingernails"),
]

def main():
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "..", "weights", "sam3-coreml")
    entries = []
    for pid, token, phrase in PROMPTS:
        enc = tok(phrase, padding="max_length", max_length=32, truncation=True)
        entries.append({"id": pid, "token": token, "phrase": phrase,
                        "tokenIds": enc["input_ids"],
                        "attentionMask": enc["attention_mask"]})
        print(f"{pid} {token}: {enc['input_ids'][:8]}…")
    path = os.path.join(out_dir, "prompt_tokens.json")
    json.dump({"tokenizer": "openai/clip-vit-base-patch32 (CLIP BPE, max_length 32)",
               "note": "validated empirically against SAM3.1 CoreML on test images — see LITO_TECHNICAL_NOTES",
               "prompts": entries}, open(path, "w"), indent=1)
    print(f"✓ wrote {path}")

if __name__ == "__main__":
    main()
