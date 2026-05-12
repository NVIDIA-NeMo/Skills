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

import argparse
import importlib.util
import json
import tempfile
import urllib.request
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

LANG_LIBS_URL = "https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/refs/heads/main/lm_eval/tasks/mmlu_prox/lang_libs.py"


def download_and_parse_lang_libs() -> tuple[dict, dict]:
    """Download lang_libs.py from GitHub (if not cached) and import it as a module."""
    # Cache the file in the same directory as this script
    cache_dir = Path(__file__).absolute().parent
    cached_file_path = cache_dir / "lang_libs.py"

    # Check if cached file exists
    if cached_file_path.exists():
        print(f"Using cached lang_libs.py from {cached_file_path}")
        lang_libs_path = str(cached_file_path)
    else:
        print(f"Downloading lang_libs.py from {LANG_LIBS_URL}...")
        try:
            with urllib.request.urlopen(LANG_LIBS_URL) as response:
                content = response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to download lang_libs.py from {LANG_LIBS_URL}: {e}")

        # Save to cache
        try:
            with open(cached_file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Cached lang_libs.py to {cached_file_path}")
            lang_libs_path = str(cached_file_path)
        except Exception as e:
            # If we can't write to cache, use a temporary file
            print(f"Warning: Could not cache file ({e}), using temporary file")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
                temp_file.write(content)
                lang_libs_path = temp_file.name

    try:
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location("lang_libs", lang_libs_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to create module spec")

        lang_libs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lang_libs_module)

        # Extract the required variables
        if not hasattr(lang_libs_module, "LANG_LIBS"):
            raise RuntimeError("LANG_LIBS not found in downloaded module")
        if not hasattr(lang_libs_module, "LANG_SUBJECTS"):
            raise RuntimeError("LANG_SUBJECTS not found in downloaded module")

        return lang_libs_module.LANG_LIBS, lang_libs_module.LANG_SUBJECTS

    except Exception as e:
        raise RuntimeError(f"Failed to import and parse lang_libs module: {e}")
    finally:
        # Clean up temporary file if we used one (not the cached file)
        if not cached_file_path.exists() and lang_libs_path != str(cached_file_path):
            Path(lang_libs_path).unlink(missing_ok=True)


def format_entry(entry: dict, language: str, lang_libs: dict, lang_subjects: dict) -> dict:
    category = entry["category"].replace(" ", "_")  # Fix computer science category

    entry["options"] = []
    for i in range(10):
        entry["options"].append(entry[f"option_{i}"])
        del entry[f"option_{i}"]

    subject = lang_subjects[language][category]
    description = lang_libs[language][3].format(subject=subject, ans_suffix=lang_libs[language][5].format("X")) + "\n"

    def get_mcq_fields(question, choices, language):
        options_dict = {chr(ord("A") + i): option for i, option in enumerate(choices)}
        options_text = "\n".join(f"{letter}. {option}" for letter, option in options_dict.items())
        question_text = f"{lang_libs[language][0]}\n{question}\n{lang_libs[language][1]}\n{options_text}\n"
        return {
            "question": question_text,
            "problem": question,  # use this field to log unwrapped question for debugging
            "options": options_text,
            **options_dict,
        }

    extract_regex = lang_libs[language][5].replace("({})", r"\(?([ABCDEFGHIJ])\)?")
    if language == "en":
        extract_regex = extract_regex.lstrip("the").strip()
        extract_regex = extract_regex.replace("\\(", "\\**\\(")
        extract_regex = extract_regex.replace("\\)?", "\\)?\\**")

    return {
        "expected_answer": entry["answer"],
        "examples_type": f"mmlu_prox_few_shot_{language}",
        "description": description,
        "extract_from_boxed": False,
        "extract_regex": extract_regex,
        "subset_for_metrics": language,
        "category": category,
        **get_mcq_fields(entry["question"], entry["options"], language),
    }


def write_data_to_file(
    output_file: Path, datasets: list, languages: list[str], lang_libs: dict, lang_subjects: dict
) -> None:
    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, dataset in enumerate(datasets):
            for entry in tqdm(dataset, desc=f"Writing {output_file.name}"):
                json.dump(
                    format_entry(
                        entry,
                        language=languages[idx],
                        lang_libs=lang_libs,
                        lang_subjects=lang_subjects,
                    ),
                    fout,
                )
                fout.write("\n")


def main(args):
    # Download and parse lang_libs data from GitHub (or use cached version)
    lang_libs, lang_subjects = download_and_parse_lang_libs()
    print("Successfully loaded lang_libs data.")

    datasets = [load_dataset("li-lab/MMLU-ProX", lang)[args.split] for lang in args.languages]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(
        output_file,
        datasets,
        languages=args.languages,
        lang_libs=lang_libs,
        lang_subjects=lang_subjects,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("validation", "test"), help="Dataset split to process.")
    parser.add_argument(
        "--languages", default=["en", "de", "es", "fr", "it", "ja"], nargs="+", help="Languages to process."
    )
    args = parser.parse_args()
    main(args)
