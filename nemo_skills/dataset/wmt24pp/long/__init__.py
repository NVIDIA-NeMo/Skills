# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Long-context translation: all documents for a language concatenated into
# one record. Tests long-input / long-output translation capability.
# Evaluation uses SEGALE (same as wmt24pp.doc).

METRICS_TYPE = "translation"
GENERATION_ARGS = "++prompt_config=multilingual/document-translation ++inference.endpoint_type=text"
GENERATION_MODULE = "nemo_skills.inference.document_translation"
NUM_CHUNKS = 11

JUDGE_PIPELINE_ARGS = {
    "judge_step_fn": "nemo_skills.pipeline.judges.segale_judge_step::create_judge_tasks",
    "segmenter": "ersatz",
    "qe_only": True,
    "save_spans": True,
    "target_languages": [
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
    ],
}
