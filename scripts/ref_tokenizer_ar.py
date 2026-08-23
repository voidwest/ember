#!/usr/bin/env python3
"""Phase 5 Track J: Arabic/multilingual tokenizer parity battery.

Tokenizes a fixed battery of Arabic (MSA + dialects), code-switched,
numeric, punctuation, and Unicode-edge strings with the reference HF
tokenizers and dumps exact token ids to JSON for comparison against
ember's EmberTokenizer.

Usage: python scripts/ref_tokenizer_ar.py <tokenizer.json> <out.json>
"""
import json
import sys

from transformers import AutoTokenizer

# The battery: every category from the Phase-5 brief that applies to text.
BATTERY = {
    # --- MSA ---
    "msa_simple": "اللغة العربية جميلة.",
    "msa_question": "ما هو الوقت الآن؟",
    "msa_long": "تُعدُّ اللغة العربية واحدة من أكثر اللغات تحدثًا في العالم، وهي اللغة الرسمية في أكثر من عشرين دولة.",
    "msa_diacritized": "الْعَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ جِدًّا.",
    "msa_tatweel": "مرحبـــــا بالعـــالم",
    # --- dialects ---
    "gulf": "شخبارك؟ وين رايح اليوم؟",
    "egyptian": "إزيك؟ عامل إيه النهارده؟",
    "levantine": "شو أخبارك؟ كيف حالك اليوم؟",
    "maghrebi": "واش راك؟ لا باس عليك؟",
    # --- code switching ---
    "codeswitch_1": "أنا أستخدم Rust للبرمجة والذكاء الاصطناعي",
    "codeswitch_2": "Let's meet غدا at 3pm في المكتب",
    "codeswitch_3": "الملف file.txt موجود في الـ folder",
    # --- numerals ---
    "arabic_indic_digits": "السنة ١٤٤٧ هجرية، والعام ٢٠٢٦ ميلادي",
    "western_digits_ar": "في عام 2026 سافر إلى الرياض.",
    "mixed_digits": "رقمي ٠٥٥٥١٢٣٤٥٦ أو 0555123456",
    "decimal_number": "السعر ١٩٫٩٩ ريالاً (19.99).",
    # --- dates / URLs / emoji / punctuation ---
    "date_ar": "تاريخ الميلاد: ١٥ رمضان ١٤٤٧هـ — 15/03/2026",
    "url_with_arabic": "https://example.com/مسار/صفحة?q=بحث",
    "emoji_mixed": "مرحباً 👋 كيف الحال؟ 🌟",
    "arabic_punct": "هل تعلم؟ نعم! لا؛ ربما… «اقتباس» [قوس] {قوس}",
    "latin_punct_in_ar": "الكلمة \"hello\" بين علامتي تنصيص، ثم فاصلة، ثم نقطة.",
    # --- unicode edges ---
    "hamza_variants": "أ إ آ ء ؤ ئ ٲ ٱ",
    "alef_variants": "ا أ إ آ ٱ اً",
    "ta_marbuta_alef": "مدرسة vs مدرسه",
    "presentation_forms": "ﻟﻐﺔ ﻋﺮﺑﻴﺔ",  # presentation forms (should not normalize)
    "combining_marks": "لغة \u0644\u0651\u064F\u063A\u064E\u0629 with shadda+damma+fatha",
    "zero_width": "عربي\u200bعربي\u200cعربي‍",
    "rtl_marks": "\u200fعربي\u200e LTR \u061c",
    "nfc_decomposed": "لُغَةٌ",  # NFD vs NFC sensitivity check happens in rust side
    "lam_alef": "لا لأ لإ لآ",
    "superscripts": "الطول ٣٠ سم² والماء H₂O",
    "currency": "السعر ٥٠ ريالاً، $50، €30",
}


def main() -> None:
    tok_path, out_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(
        tok_path, local_files_only=True, cache_dir=None
    )
    out = {"tokenizer": tok_path, "strings": {}}
    for key, text in BATTERY.items():
        ids = tok.encode(text)
        tokens = tok.convert_ids_to_tokens(ids)
        out["strings"][key] = {
            "text": text,
            "ids": ids,
            "tokens": tokens,
        }
        print(f"{key}: {len(ids)} tokens")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
