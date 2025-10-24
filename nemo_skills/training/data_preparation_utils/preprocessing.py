# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from typing import Dict, Optional

from sdp.processors.base_processor import BaseProcessor
from tqdm.contrib.concurrent import process_map

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.prompt.utils import get_prompt
from nemo_skills.utils import get_logger_name, unroll_files

LOG = logging.getLogger(get_logger_name(__file__))


class ReadData(BaseProcessor):
    def __init__(
        self,
        input_files: Optional[str] = None,
        preprocessed_dataset_files: Optional[str] = None,
        input_key="question",
        output_key=None,  # if None, we are not doing correctness checks
        skip_first: int = 0,
        add_correct: bool = True,
        add_incorrect: bool = False,
        add_unlabeled: bool = False,
        use_judgement: bool = False,
        keys_to_keep: list[str] | None = None,
        deduplicate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_files = input_files
        self.preprocessed_dataset_files = preprocessed_dataset_files
        self.input_key = input_key
        self.output_key = output_key
        self.skip_first = skip_first
        self.add_correct = add_correct
        self.add_incorrect = add_incorrect
        self.add_unlabeled = add_unlabeled
        self.use_judgement = use_judgement
        self.keys_to_keep = keys_to_keep
        self.deduplicate = deduplicate

        if self.keys_to_keep is not None:
            self.keys_to_keep = set(self.keys_to_keep)
            self.keys_to_keep.add(self.input_key)
            if self.output_key is not None:
                self.keys_to_keep.add(self.output_key)
                self.keys_to_keep.add("symbolic_correct")
                self.keys_to_keep.add("judgement")
                self.keys_to_keep.add("reasoning_content")

        if isinstance(self.input_files, str):
            if "," in self.input_files:
                self.input_files = self.input_files.split(",")
            else:
                self.input_files = self.input_files.split(" ")

        if isinstance(self.preprocessed_dataset_files, str):
            if "," in self.preprocessed_dataset_files:
                self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(",")
            else:
                self.preprocessed_dataset_files = self.preprocessed_dataset_files.split(" ")

        if self.input_files is None and self.preprocessed_dataset_files is None:
            raise ValueError("Either `input_files` or `preprocessed_dataset_files` should be provided")

        if self.output_key is not None and not self.add_correct and not self.add_incorrect:
            raise ValueError("At least one of `add_correct` and `add_incorrect` should be True")

    def _read_preprocessed_data(self, file_handle) -> int:
        samples = []
        questions = set()
        for idx, line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
            sample = json.loads(line)
            if self.keys_to_keep:
                sample = {k: v for k, v in sample.items() if k in self.keys_to_keep}
            questions.add(sample[self.input_key])
            samples.append(sample)

        return samples

    def _parallel_read_file(self, args):
        file_path, read_fn = args
        with open(file_path, "rt", encoding="utf-8") as file_handle:
            samples = read_fn(file_handle)
        return samples

    def _read_raw_data(self, file_handle) -> int:
        samples = []

        for idx, file_line in enumerate(file_handle):
            if idx < self.skip_first:
                continue
            # if different files have different number of lines
            if file_line is None:
                continue
            line_dict = json.loads(file_line)
            # can be empty for incomplete generations
            if not line_dict:
                continue

            if self.output_key is not None and not self.add_unlabeled:
                if not self.use_judgement:
                    if "symbolic_correct" not in line_dict:
                        LOG.warning("Found incomplete generations (symbolic_correct field is missing) - skipping")
                        continue

                    if not self.add_correct and line_dict["symbolic_correct"]:
                        continue

                    if not self.add_incorrect and not line_dict["symbolic_correct"]:
                        continue
                else:
                    if "judgement" not in line_dict:
                        LOG.warning("Found incomplete generations (judgement field is missing) - skipping")
                        continue

                    if not self.add_correct and is_correct_judgement(line_dict["judgement"]):
                        continue

                    if not self.add_incorrect and not is_correct_judgement(line_dict["judgement"]):
                        continue

            line_dict["filename"] = file_handle.name

            if self.keys_to_keep:
                line_dict = {k: v for k, v in line_dict.items() if k in self.keys_to_keep}

            samples.append(line_dict)

        return samples

    def _get_sample_hash(self, sample):
        if self.output_key is None:
            return sample[self.input_key]
        return (sample[self.input_key], sample[self.output_key])

    def _batch_deduplicate(self, batch):
        seen_predictions = set()
        unique_samples = []

        for sample in batch:
            sample_hash = self._get_sample_hash(sample)
            if sample_hash not in seen_predictions:
                seen_predictions.add(sample_hash)
                unique_samples.append(sample)

        return unique_samples

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def process(self):
        samples = []
        if self.input_files:
            args = [(file, self._read_raw_data) for file in unroll_files(self.input_files)]
            results = process_map(self._parallel_read_file, args, max_workers=4, chunksize=1)
            samples.extend(list(chain(*results)))
        if self.preprocessed_dataset_files:
            args = [(file, self._read_preprocessed_data) for file in unroll_files(self.preprocessed_dataset_files)]
            results = process_map(self._parallel_read_file, args, max_workers=None, chunksize=1)
            samples.extend(list(chain(*results)))

        if self.deduplicate:
            LOG.info("Total samples before deduplication: %d", len(samples))

            # Parallel deduplication
            chunk_size = 100000
            num_cores = max(100, len(samples) // chunk_size)

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Process chunks in parallel
                futures = [
                    executor.submit(self._batch_deduplicate, chunk) for chunk in self._chunks(samples, chunk_size)
                ]

                # Final deduplication of results from all chunks
                seen_predictions = set()
                samples_count = 0

                with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
                    for future in futures:
                        chunk_results = future.result()
                        for sample in chunk_results:
                            sample_hash = self._get_sample_hash(sample)
                            if sample_hash not in seen_predictions:
                                seen_predictions.add(sample_hash)
                                fout.write(json.dumps(sample) + "\n")
                                samples_count += 1

            LOG.info("Total samples after deduplication: %d", samples_count)
        else:
            LOG.info("Total samples: %d", len(samples))
            with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
                for sample in samples:
                    fout.write(json.dumps(sample) + "\n")


class GroupSamples(BaseProcessor):
    def __init__(self, group_key="input", **kwargs):
        super().__init__(**kwargs)
        self.group_key = group_key

    def process(self):
        samples = defaultdict(list)
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                sample = json.loads(line)
                samples[sample[self.group_key]].append(sample)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for groupped_samples in samples.values():
                fout.write(json.dumps(groupped_samples) + "\n")


class ShuffleAndDownsampleData(BaseProcessor):
    def __init__(
        self,
        random_seed: int,
        do_shuffle: bool,
        num_samples: Optional[int] = None,
        sampling_method: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampling_method = sampling_method
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.do_shuffle = do_shuffle

        if self.sampling_method not in [None, "random", "fair"]:
            raise ValueError(
                f"Sampling method {self.sampling_method} is not supported, use `None`, `random` or `fair`"
            )

        if self.sampling_method is None and self.num_samples is not None:
            raise ValueError("Number of samples can be specified only when sampling method is `random` or `fair`")

        if self.sampling_method is not None and self.num_samples is None:
            raise ValueError("Number of samples should be specified when sampling method is `random` or `fair`")

    def process(self):
        groupped_samples = []
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                samples = json.loads(line)
                groupped_samples.append(samples)

        random.seed(self.random_seed)
        if self.sampling_method is None:
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
        if self.sampling_method == "random":
            output_instances = list(chain(*groupped_samples))
            if self.do_shuffle:
                random.shuffle(output_instances)
            output_instances = output_instances[: self.num_samples]
        elif self.sampling_method == "fair":
            soln_counter = 0
            output_instances = []
            num_input_samples = sum(len(samples) for samples in groupped_samples)
            if num_input_samples < self.num_samples:
                LOG.warning(
                    "Total SFT entries %d is not less than `num_output_samples` %d, skipping downsampling.",
                    num_input_samples,
                    self.num_samples,
                )
                output_instances = list(chain(*groupped_samples))
            # downsample only if num_input_samples > self.num_samples
            while len(output_instances) < self.num_samples and num_input_samples > self.num_samples:
                for quesn_idx in range(len(groupped_samples)):
                    if len(output_instances) == self.num_samples:
                        break
                    if len(groupped_samples[quesn_idx]) > soln_counter:
                        output_instances.append(groupped_samples[quesn_idx][soln_counter])
                soln_counter += 1
            if self.do_shuffle:
                random.shuffle(output_instances)

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for instance in output_instances:
                fout.write(json.dumps(instance) + "\n")


class WriteFinalSftManifest(BaseProcessor):
    def __init__(
        self,
        prompt_config: str,
        tokenizer: str | None = None,
        chat_template_kwargs: dict | None = None,
        system_message: str | None = None,
        assistant_end: str | None = None,
        code_tags: str | None = None,
        input_key: str = "input",
        output_key: str = "output",
        metadata: Optional[Dict] = None,
        exclude_optional_keys: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.metadata = metadata
        self.exclude_optional_keys = exclude_optional_keys
        if not self.metadata:
            self.metadata = {}

        self.prompt = None
        if prompt_config and tokenizer:
            self.prompt = get_prompt(
                prompt_config, tokenizer=tokenizer, system_message=system_message, code_tags=code_tags
            )
        else:
            if tokenizer:
                LOG.warning(
                    "tokenizer is provided, but prompt config is missing! "
                    "Assuming 'user: {input_key}' and no special formatting for output."
                )
                self.prompt = get_prompt(
                    {"user": "{" + input_key + "}"},
                    tokenizer=tokenizer,
                    system_message=system_message,
                    code_tags=code_tags,
                )
            else:
                LOG.warning("Prompt details are missing! The processed data won't be formatted using any prompt.")

        if system_message is not None:
            if prompt_config is None or tokenizer is None:
                raise ValueError("prompt_config and tokenizer are required when system_message is provided")

        self.chat_template_kwargs = chat_template_kwargs
        self.assistant_end = assistant_end

    def process(self):
        samples_count = 0
        seen_predictions = defaultdict(set)
        with (
            open(self.input_manifest_file, "rt", encoding="utf-8") as fin,
            open(self.output_manifest_file, "wt", encoding="utf-8") as fout,
        ):
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)
                question = elem[self.input_key]
                # deduplication
                if elem[self.output_key] in seen_predictions[question]:
                    continue
                seen_predictions[question].add(elem[self.output_key])
                if "expected_answer" in elem:
                    elem["expected_answer"] = str(elem["expected_answer"])
                # take only required keys from the input if exclude_optional_keys is True
                output_sample = {}
                if not self.exclude_optional_keys:
                    output_sample = json.loads(line)
                elif "expected_answer" in elem:
                    output_sample["expected_answer"] = elem["expected_answer"]

                generation = elem.pop(self.output_key)
                if self.prompt:
                    output_sample["input"] = self.prompt.fill(
                        input_dict=elem, chat_template_kwargs=self.chat_template_kwargs, format_as_string=True
                    )
                    # not adding end-of-turn for incomplete generations
                    if output_sample.get("finish_reason", "stop") == "stop":
                        if self.assistant_end is None:
                            output_sample["output"] = self.prompt.format_assistant_response(
                                content=generation,
                                thinking=elem.get("reasoning_content"),
                                chat_template_kwargs=self.chat_template_kwargs,
                            )
                        else:
                            if elem.get("reasoning_content"):
                                raise ValueError("reasoning_content is not supported with assistant_end parameter")
                            output_sample["output"] = generation + self.assistant_end
                    else:
                        # this doesn't work properly with reasoning_content, so let's fail for now. TODO: fix this
                        if elem.get("reasoning_content"):
                            raise ValueError("reasoning_content is not supported yet for incomplete generations")
                        output_sample["output"] = generation

                else:
                    output_sample["input"] = elem[self.input_key]
                    output_sample["output"] = generation

                output_sample.update(self.metadata)
                fout.write(json.dumps(output_sample) + "\n")
                samples_count += 1

        LOG.info("Prepared dataset size: %d", samples_count)


class WriteFinalRLManifest(BaseProcessor):
    def __init__(
        self,
        prompt_config: str,
        tokenizer: str | None = None,
        chat_template_kwargs: dict | None = None,
        system_message: str | None = None,
        code_tags: str | None = None,
        task_name: str | None = None,
        input_key: str = "input",
        metadata: dict | None = None,
        exclude_optional_keys: bool = True,
        random_seed: int = 0,
        do_shuffle: bool = True,
        num_output_samples: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_key = input_key
        self.metadata = metadata
        self.exclude_optional_keys = exclude_optional_keys
        if not self.metadata:
            self.metadata = {}

        self.prompt = None
        if prompt_config and tokenizer:
            self.prompt = get_prompt(
                prompt_config, tokenizer=tokenizer, system_message=system_message, code_tags=code_tags
            )
        else:
            if tokenizer:
                LOG.warning(
                    "tokenizer is provided, but prompt config is missing! "
                    "Assuming 'user: {input_key}' and no special formatting for output."
                )
                self.prompt = get_prompt({"user": "{" + input_key + "}"}, tokenizer=tokenizer, code_tags=code_tags)
            else:
                LOG.warning("Prompt details are missing! The processed data won't be formatted using any prompt.")

        if system_message is not None:
            if prompt_config is None or tokenizer is None:
                raise ValueError("prompt_config and tokenizer are required when system_message is provided")

        self.chat_template_kwargs = chat_template_kwargs
        self.random_seed = random_seed
        self.do_shuffle = do_shuffle
        self.num_output_samples = num_output_samples
        self.task_name = task_name

    def process(self):
        samples_count = 0
        all_data = []
        with open(self.input_manifest_file, "rt", encoding="utf-8") as fin:
            # only looping over the correct samples (unless asked for incorrect)
            for line in fin:
                elem = json.loads(line)
                if "expected_answer" in elem:
                    elem["expected_answer"] = str(elem["expected_answer"])
                # take only required keys from the input if exclude_optional_keys is True
                output_sample = {}
                if not self.exclude_optional_keys:
                    output_sample = json.loads(line)
                else:
                    # including only required keys if they are present
                    if "expected_answer" in elem:
                        output_sample["expected_answer"] = elem["expected_answer"]
                    if "problem" in elem:
                        output_sample["problem"] = elem["problem"]

                if self.prompt:
                    output_sample["input"] = self.prompt.fill(
                        input_dict=elem, chat_template_kwargs=self.chat_template_kwargs
                    )
                else:
                    output_sample["input"] = elem[self.input_key]

                if self.task_name:
                    output_sample["task_name"] = self.task_name

                output_sample.update(self.metadata)
                all_data.append(output_sample)
                samples_count += 1

        LOG.info("Full dataset size: %d", samples_count)

        if self.do_shuffle:
            random.seed(self.random_seed)
            random.shuffle(all_data)

        if self.num_output_samples is not None:
            all_data = all_data[: self.num_output_samples]
            LOG.info("Downsampled dataset size: %d", len(all_data))

        with open(self.output_manifest_file, "wt", encoding="utf-8") as fout:
            for sample in all_data:
                fout.write(json.dumps(sample) + "\n")
