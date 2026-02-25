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

import json
import logging
from dataclasses import field
from pathlib import Path
from typing import List

import hydra
import iso639

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.utils import nested_dataclass, setup_logging

LOG = logging.getLogger(__name__)


def is_line_translatable_content(line: str) -> bool:
    """Determines if the content of a line is considered translatable."""
    stripped_line = line.strip()
    if not any(char.isalpha() for char in stripped_line):
        return False
    if stripped_line.startswith("<") and stripped_line.endswith(">"):
        return False
    return True


def _get_all_nested_fields(obj, path: str) -> List[str]:
    """A helper function that finds all string values for a field specified by `path`."""
    parts = path.split(".")

    def finder(current_obj, current_path, collected_values):
        if not current_path:
            return

        key = current_path[0]
        rest_of_path = current_path[1:]

        if key == "*":
            if isinstance(current_obj, list):
                for item in current_obj:
                    if not rest_of_path:
                        if isinstance(item, str):
                            collected_values.append(item)
                    else:
                        finder(item, rest_of_path, collected_values)
            return

        if isinstance(current_obj, dict) and key in current_obj:
            if not rest_of_path:
                value = current_obj[key]
                if isinstance(value, str):
                    collected_values.append(value)
            else:
                finder(current_obj[key], rest_of_path, collected_values)

    found_values = []
    finder(obj, parts, found_values)
    return found_values


def full_language_name(lang_code: str) -> str:
    """
    Convert language code to full language name using iso639.

    Args:
        lang_code: ISO 639-1 (2-letter) language code

    Returns:
        Full English name of the language, or original code if not found
    """

    lang_code = lang_code.lower().strip()
    language = iso639.Lang(lang_code)
    if language:
        return language.name

    return lang_code


@nested_dataclass(kw_only=True)
class TranslationConfig(GenerationTaskConfig):
    target_lang: str = "zh"
    use_skipme: bool = False
    fields_to_translate: List[str] = field(default_factory=lambda: ["conversations.*.value"])
    translation_key: str = "translations"

    # Override inference defaults for translation tasks to match with ported settings
    inference: InferenceConfig = field(
        default_factory=lambda: InferenceConfig(
            top_k=-1,
            temperature=0.7,
            top_p=0.8,
        )
    )


class TranslationTask(GenerationTask):
    def __init__(self, cfg: TranslationConfig):
        super().__init__(cfg)
        self.cfg = cfg

    def preprocess_data(self, data):
        """
        Transform loaded JSONL data following the original translate_multiline_doc logic.
        Processes documents line-by-line, handles code blocks, and extracts translatable content.

        Args:
            data: List of original data points from load_data()

        Returns:
            List of translation data points with only the core fields:
            [{'source_lang': 'English', 'target_lang': 'Hebrew', 'src': 'text'}, ...]
        """
        LOG.info(f"Preprocessing {len(data)} data points for translation")

        # Filter out skipme entries
        original_dps = [dp for dp in data if not (self.cfg.use_skipme and "_skipme" in dp and dp["_skipme"] != 0)]
        LOG.info(f"After filtering skipme: {len(original_dps)} data points")

        # Extract all text content that needs translation
        all_docs = []
        dp_info_for_translation = []

        for dp in original_dps:
            current_dp_info = []
            for field_path in self.cfg.fields_to_translate:
                has_wildcard = "*" in field_path
                path_parts = field_path.split(".")
                field_key = path_parts[-1]  # Always use the last part (e.g., "value" from "conversations.*.value")
                texts = _get_all_nested_fields(dp, field_path)
                for text in texts:
                    current_dp_info.append((field_key, text, has_wildcard))
                    all_docs.append(text)
            dp_info_for_translation.append(current_dp_info)

        # Before we translate, we need to extract all the lines that require translation from the documents.
        # However, it's not enough to just extract the translatable lines: we also need to remember the structure and formatting of the original documents,
        # so that after translation, we can reconstruct the documents with the translated lines in their correct places, preserving code blocks, indentation, and other formatting.
        #
        # The following variables are used to store this information:
        # - all_translatable_lines: a flat list of all lines that need translation, across all documents, in order.
        # - doc_boundaries: indices marking where each document's translatable lines start/end in all_translatable_lines.
        # - doc_templates: for each non-translatable line, we store the original line (they will be carried over as-is to the output), otherwise, we store None as placeholder.
        # - doc_translatable_indices: for each document, the indices of lines (within the document) that are to be translated.
        # - doc_leading_spaces_lists: this is an extension of doc_templates -- this stores the leading whitespace for each translatable line, so we can restore indentation.
        # - doc_original_stripped_lines_for_segmentation: for each document, the original stripped lines to help with segmentation or reference.
        #
        # All of this information will be used in the postprocess function to reconstruct the original documents with translated content, preserving formatting.

        all_translatable_lines = []
        doc_boundaries = [0]
        doc_templates = []
        doc_translatable_indices = []
        doc_leading_spaces_lists = []
        doc_original_stripped_lines_for_segmentation = []

        for doc_content in all_docs:
            lines = doc_content.split("\n")
            current_doc_template = []
            current_doc_translatable_lines = []
            current_doc_translatable_indices = []
            current_doc_leading_spaces_list = []
            current_doc_original_stripped_lines = []
            in_code_block = False

            for i, line in enumerate(lines):
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    current_doc_template.append(line)
                    continue

                if in_code_block or not is_line_translatable_content(line):
                    current_doc_template.append(line)
                else:
                    num_leading_spaces = len(line) - len(line.lstrip())
                    leading_spaces = line[:num_leading_spaces]
                    stripped_line = line[num_leading_spaces:]

                    current_doc_template.append(None)
                    current_doc_translatable_lines.append(stripped_line)
                    all_translatable_lines.append(stripped_line)
                    current_doc_translatable_indices.append(i)
                    current_doc_leading_spaces_list.append(leading_spaces)
                    current_doc_original_stripped_lines.append(stripped_line)

            doc_templates.append(current_doc_template)
            doc_translatable_indices.append(current_doc_translatable_indices)
            doc_leading_spaces_lists.append(current_doc_leading_spaces_list)
            doc_boundaries.append(len(all_translatable_lines))
            doc_original_stripped_lines_for_segmentation.append(current_doc_original_stripped_lines)

        # Store reconstruction data for postprocessing
        self._reconstruction_data = {
            "original_dps": original_dps,
            "dp_info_for_translation": dp_info_for_translation,
            "doc_boundaries": doc_boundaries,
            "doc_templates": doc_templates,
            "doc_translatable_indices": doc_translatable_indices,
            "doc_leading_spaces_lists": doc_leading_spaces_lists,
            "doc_original_stripped_lines_for_segmentation": doc_original_stripped_lines_for_segmentation,
        }

        # Create translation data points with only the core fields
        # We need to preserve the _async_position key for proper order restoration
        translation_data_points = []
        async_position = 0
        for text in all_translatable_lines:
            translation_data_point = {
                "source_lang": full_language_name("en"),
                "target_lang": full_language_name(self.cfg.target_lang),
                "src": text,
                self.cfg.async_position_key: async_position,
            }
            translation_data_points.append(translation_data_point)
            async_position += 1

        LOG.info(f"Created {len(translation_data_points)} individual translation tasks")
        return translation_data_points

    def log_example_prompt(self, data):
        """Override log_example_prompt to work with translation data format."""
        if data:
            # Use the first translation data point for logging
            sample_data_point = (
                data[0]
                if isinstance(data[0], dict) and "src" in data[0]
                else {
                    "source_lang": "English",
                    "target_lang": full_language_name(self.cfg.target_lang),
                    "src": "Sample text to translate",
                }
            )
            filled_prompt = self.fill_prompt(sample_data_point, data)
            LOG.info("Example prompt:\nData dictionary: %s\nPrompt: %s", sample_data_point, filled_prompt)
        else:
            LOG.info("No data available for example prompt")

    def unwrap_translation_results(self, text):
        """
        Unwrap the translation results from the generated output. The format defined here is 〘...〙.

        If the result cannot be extracted, return the original text.
        """
        left_loc = text.rfind("〘")
        right_loc = text.rfind("〙")
        if left_loc != -1 and right_loc != -1 and left_loc < right_loc:
            text = text[left_loc + 1 : right_loc]
        elif left_loc != -1:
            text = text[left_loc + 1 :]
        else:
            pass  # do nothing

        return text

    def postprocess(self):
        """
        Reassemble individual translation results back into original document structure.
        Reads the output file written by base class, reconstructs documents, and writes back.
        """

        LOG.info("Starting postprocessing: reassembling translations into original document structure")

        # Read the individual translation results written by base class
        output_file = Path(self.cfg.output_file)
        if not output_file.exists():
            LOG.warning(f"Output file {output_file} does not exist, skipping postprocessing")
            return

        # Load individual translation results
        individual_results = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    individual_results.append(json.loads(line))

        LOG.info(f"Loaded {len(individual_results)} individual translation results")

        # Extract translated texts from results
        translated_segments_for_all_docs = []
        for result in individual_results:
            # The base class should have stored the generation in 'generation' field
            if "generation" in result:
                translated_segments_for_all_docs.append(result["generation"])
            elif "response" in result:
                translated_segments_for_all_docs.append(result["response"])
            else:
                # Fallback: look for any text field
                translated_segments_for_all_docs.append(str(result))

        reconstruction_data = self._reconstruction_data
        original_dps = reconstruction_data["original_dps"]
        dp_info_for_translation = reconstruction_data["dp_info_for_translation"]
        doc_boundaries = reconstruction_data["doc_boundaries"]
        doc_templates = reconstruction_data["doc_templates"]
        doc_translatable_indices = reconstruction_data["doc_translatable_indices"]
        doc_leading_spaces_lists = reconstruction_data["doc_leading_spaces_lists"]
        doc_original_stripped_lines_for_segmentation = reconstruction_data[
            "doc_original_stripped_lines_for_segmentation"
        ]

        # Reassemble documents
        all_results = []
        all_segmented_translations = []

        for doc_idx in range(len(doc_templates)):
            template = doc_templates[doc_idx].copy()  # Make a copy to avoid modifying original
            translatable_indices = doc_translatable_indices[doc_idx]
            leading_spaces_list = doc_leading_spaces_lists[doc_idx]
            original_stripped_lines_for_doc = doc_original_stripped_lines_for_segmentation[doc_idx]

            start_offset = doc_boundaries[doc_idx]
            end_offset = doc_boundaries[doc_idx + 1]
            doc_translated_segments = translated_segments_for_all_docs[start_offset:end_offset]
            current_doc_segmented_translations = []

            if not translatable_indices:
                all_results.append("\n".join(template))
                all_segmented_translations.append([])
                continue

            for i, original_stripped_line in enumerate(original_stripped_lines_for_doc):
                idx_in_template = translatable_indices[i]
                leading_spaces = leading_spaces_list[i]
                translated_segment = self.unwrap_translation_results(doc_translated_segments[i]).lstrip()
                current_doc_segmented_translations.append({"src": original_stripped_line, "tgt": translated_segment})
                template[idx_in_template] = leading_spaces + translated_segment

            all_results.append("\n".join(template))
            all_segmented_translations.append(current_doc_segmented_translations)

        final_output_data = []
        translation_idx = 0

        # Calculate expected number of documents from dp_info_for_translation
        expected_docs = sum(len(dp_info) for dp_info in dp_info_for_translation)
        if len(all_results) == expected_docs:
            for i, dp in enumerate(original_dps):
                output_dp = dp.copy()
                dp_info = dp_info_for_translation[i]
                if not dp_info:
                    final_output_data.append(output_dp)
                    continue

                dp_translations_map, dp_segmented_map = {}, {}
                for field_key, _, has_wildcard in dp_info:
                    if translation_idx < len(all_results):
                        translated_text = all_results[translation_idx]
                        segmented_list = all_segmented_translations[translation_idx]
                        translation_idx += 1

                        if has_wildcard:
                            dp_translations_map.setdefault(field_key, []).append(translated_text)
                            dp_segmented_map.setdefault(field_key, []).extend(segmented_list)
                        else:
                            dp_translations_map[field_key] = translated_text
                            dp_segmented_map[field_key] = segmented_list

                output_dp[self.cfg.translation_key] = {
                    "target_lang": self.cfg.target_lang,
                    "translation": dp_translations_map,
                    "segmented_translation": dp_segmented_map,
                }
                final_output_data.append(output_dp)
        else:
            LOG.error(f"The input has {expected_docs} documents, but the output has {len(all_results)} documents.")
            final_output_data = original_dps

        # Write reassembled results back to output file
        LOG.info(f"Writing {len(final_output_data)} reassembled documents to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            for dp in final_output_data:
                f.write(json.dumps(dp, ensure_ascii=False) + "\n")

        LOG.info("Postprocessing completed successfully")


GENERATION_TASK_CLASS = TranslationTask

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_translation_config", node=TranslationConfig)


@hydra.main(version_base=None, config_name="base_translation_config")
def main(cfg: TranslationConfig):
    cfg = TranslationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = TranslationTask(cfg)
    task.generate()


if __name__ == "__main__":
    setup_logging()
    main()
