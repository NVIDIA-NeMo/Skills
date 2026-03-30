# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare wmt24pp data for all sub-benchmarks (sent, doc, long).

Downloads from google/wmt24pp and writes:
  sent/<split>.jsonl  — one record per sentence (sentence-level translation)
  doc/<split>.jsonl   — one record per (document, language) (document-level translation)
  long/<split>.jsonl  — one record per language (long-context translation)
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from datasets import load_dataset
from langcodes import Language


def _lang_name(lang_code: str) -> str:
    """Convert a language code like 'fil_PH' to a display name."""
    return Language(lang_code.split("_")[0]).display_name()


def write_sent(output_file, datasets, tgt_languages):
    """One record per sentence — sentence-level translation."""
    records = []
    for tgt_lang in tgt_languages:
        for row in datasets[tgt_lang]:
            if row["is_bad_source"]:
                continue
            records.append(
                {
                    "text": row["source"],
                    "translation": row["target"],
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": _lang_name(tgt_lang),
                    "doc_id": row["document_id"],
                    "seg_id": row["segment_id"],
                }
            )
    with open(output_file, "wt", encoding="utf-8") as fout:
        for record in records:
            json.dump(record, fout)
            fout.write("\n")


def write_doc(output_file, datasets, tgt_languages):
    """One record per (document, language) — document-level translation."""
    records = []
    for tgt_lang in tgt_languages:
        docs = OrderedDict()
        for row in datasets[tgt_lang]:
            if row["is_bad_source"]:
                continue
            doc_id = row["document_id"]
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(row)

        for doc_id in docs:
            docs[doc_id].sort(key=lambda r: r["segment_id"])

        for doc_id, rows in docs.items():
            source_sentences = [row["source"] for row in rows]
            reference_sentences = [row["target"] for row in rows]
            records.append(
                {
                    "text": " ".join(source_sentences),
                    "source_sentences": source_sentences,
                    "reference_sentences": reference_sentences,
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": _lang_name(tgt_lang),
                    "doc_id": doc_id,
                    "seg_id": 1,
                }
            )
    with open(output_file, "wt", encoding="utf-8") as fout:
        for record in records:
            json.dump(record, fout)
            fout.write("\n")


def write_long(output_file, datasets, tgt_languages):
    """One record per language — long-context translation."""
    records = []
    for tgt_lang in tgt_languages:
        docs = OrderedDict()
        for row in datasets[tgt_lang]:
            if row["is_bad_source"]:
                continue
            doc_id = row["document_id"]
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(row)

        for doc_id in docs:
            docs[doc_id].sort(key=lambda r: r["segment_id"])

        doc_groups = []
        source_sentences = []
        reference_sentences = []
        for doc_id, rows in docs.items():
            group_src = [row["source"] for row in rows]
            group_ref = [row["target"] for row in rows]
            doc_groups.append({"doc_id": doc_id, "source_sentences": group_src, "reference_sentences": group_ref})
            source_sentences.extend(group_src)
            reference_sentences.extend(group_ref)

        records.append(
            {
                "text": " ".join(source_sentences),
                "source_sentences": source_sentences,
                "reference_sentences": reference_sentences,
                # doc_groups preserves per-document boundaries so that document order
                # can be shuffled at inference time for multi-seed runs.
                "doc_groups": doc_groups,
                "source_language": "en",
                "target_language": tgt_lang,
                "source_lang_name": "English",
                "target_lang_name": _lang_name(tgt_lang),
                "doc_id": tgt_lang,
                "seg_id": 1,
            }
        )
    with open(output_file, "wt", encoding="utf-8") as fout:
        for record in records:
            json.dump(record, fout)
            fout.write("\n")


SENT_DEFAULT_LANGUAGES = ["de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP"]

ALL_LANGUAGES = [
    "ar_EG",
    "ar_SA",
    "bg_BG",
    "bn_IN",
    "ca_ES",
    "cs_CZ",
    "da_DK",
    "de_DE",
    "el_GR",
    "es_MX",
    "et_EE",
    "fa_IR",
    "fi_FI",
    "fil_PH",
    "fr_CA",
    "fr_FR",
    "gu_IN",
    "he_IL",
    "hi_IN",
    "hr_HR",
    "hu_HU",
    "id_ID",
    "is_IS",
    "it_IT",
    "ja_JP",
    "kn_IN",
    "ko_KR",
    "lt_LT",
    "lv_LV",
    "ml_IN",
    "mr_IN",
    "nl_NL",
    "no_NO",
    "pa_IN",
    "pl_PL",
    "pt_BR",
    "pt_PT",
    "ro_RO",
    "ru_RU",
    "sk_SK",
    "sl_SI",
    "sr_RS",
    "sv_SE",
    "sw_KE",
    "sw_TZ",
    "ta_IN",
    "te_IN",
    "th_TH",
    "tr_TR",
    "uk_UA",
    "ur_PK",
    "vi_VN",
    "zh_CN",
    "zh_TW",
    "zu_ZA",
]


def main(args):
    data_dir = Path(__file__).absolute().parent

    for sub, writer, languages in [
        ("sent", write_sent, args.sent_languages),
        ("doc", write_doc, args.target_languages),
        ("long", write_long, args.target_languages),
    ]:
        # Download only the languages needed for this sub-benchmark.
        datasets = {}
        for lang in languages:
            if lang not in datasets:
                datasets[lang] = load_dataset("google/wmt24pp", f"en-{lang}")["train"]

        sub_dir = data_dir / sub
        sub_dir.mkdir(exist_ok=True)
        output_file = sub_dir / f"{args.split}.jsonl"
        writer(output_file, datasets, tgt_languages=languages)
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    parser.add_argument(
        "--target_languages",
        default=ALL_LANGUAGES,
        nargs="+",
        help="Languages for doc and long sub-benchmarks (default: all 55).",
    )
    parser.add_argument(
        "--sent_languages",
        default=SENT_DEFAULT_LANGUAGES,
        nargs="+",
        help="Languages for sent sub-benchmark (default: 5, preserving original wmt24pp behavior).",
    )
    args = parser.parse_args()
    main(args)
