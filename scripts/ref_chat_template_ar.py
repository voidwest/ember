#!/usr/bin/env python3
"""Phase 5 Track J3: chat-template parity for Arabic conversations.

Renders an Arabic multiturn conversation through the Llama-3.2 chat
template (the one ember's VoiceSession scaffolds by hand) and dumps the
exact token stream. Ember must reproduce it id-for-id.

Usage: python scripts/ref_chat_template_ar.py <tokenizer_dir> <out.json>
"""
import json
import sys

from transformers import AutoTokenizer

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "ما هو الوقت الآن؟"},
    {"role": "assistant", "content": "الساعة الثالثة عصرًا بتوقيت الرياض."},
    {"role": "user", "content": "شخبارك؟ وين رايح اليوم؟"},
    {"role": "assistant", "content": "أنا بخير، سأذهب إلى المكتب."},
    {"role": "user", "content": "Let's switch to English ثم ترجع للعربية، تمام?"},
]


def main() -> None:
    tok_dir, out_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    # date_string pins the template's "Today Date" line to the value
    # ember's scaffold hardcodes. tokenize=True lets the template machinery
    # handle special tokens; encoding the rendered string AGAIN would
    # double-add BOS (the template already emits <|begin_of_text|>).
    text = tok.apply_chat_template(
        CONVERSATION,
        tokenize=False,
        add_generation_prompt=True,
        date_string="01 Jan 2026",
    )
    ids = tok.encode(text, add_special_tokens=False)
    # sanity: exactly one BOS
    assert ids.count(tok.bos_token_id) == 1, "reference must contain exactly one BOS"
    out = {
        "tokenizer": tok_dir,
        "rendered": text,
        "ids": ids,
        "conversation": CONVERSATION,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("rendered chars:", len(text), "ids:", len(ids))
    print(text[:300])
    print("wrote", out_path)


if __name__ == "__main__":
    main()
