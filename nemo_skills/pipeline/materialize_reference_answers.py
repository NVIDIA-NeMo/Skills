# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import json
from pathlib import Path


def materialize_reference_answers(
    input_file: str | Path,
    output_file: str | Path,
    reference_answer_key: str,
) -> None:
    """Copy dataset rows while exposing their reference answer as generation."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_file.with_suffix(output_file.suffix + ".tmp")

    try:
        with (
            input_file.open(encoding="utf-8") as input_stream,
            temporary_output.open("w", encoding="utf-8") as output_stream,
        ):
            for line_number, line in enumerate(input_stream, start=1):
                data_point = json.loads(line)
                if reference_answer_key not in data_point:
                    raise ValueError(
                        f"Missing reference answer key {reference_answer_key!r} in {input_file} at line {line_number}"
                    )
                reference_answer = data_point[reference_answer_key]
                if not isinstance(reference_answer, str):
                    raise ValueError(
                        f"Reference answer key {reference_answer_key!r} must contain a string "
                        f"in {input_file} at line {line_number}"
                    )
                data_point["generation"] = reference_answer
                output_stream.write(json.dumps(data_point, ensure_ascii=False) + "\n")
        temporary_output.replace(output_file)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize reference answers as model generations.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--reference-answer-key", required=True)
    args = parser.parse_args()
    materialize_reference_answers(args.input_file, args.output_file, args.reference_answer_key)


if __name__ == "__main__":
    main()
