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

import argparse
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


def read_excel_to_text(excel_path: Path) -> str:
    """Read Excel file and convert to text representation."""
    import pandas as pd

    try:
        # Explicitly handle .xlsb files with pyxlsb engine
        engine = "pyxlsb" if excel_path.suffix == ".xlsb" else None
        with pd.ExcelFile(excel_path, engine=engine) as xls:
            sheets = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file {excel_path}: {e}") from e

    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = df.to_string(index=False)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text


def format_paths_for_prompt(paths: list[Path], actual_root: Path, display_root: Path) -> str:
    """Format file paths for display in prompt.

    Args:
        paths: List of absolute Path objects to format
        actual_root: Root directory where files actually exist
        display_root: Root directory to display in paths (absolute for abs paths, Path(".") for relative)
    """
    if not paths:
        return ""

    formatted = []
    for path in paths:
        try:
            rel = path.relative_to(actual_root)
            disp_path = display_root / rel
        except ValueError:
            disp_path = path
        formatted.append(str(disp_path))

    return " ".join(formatted)


def save_data(split: str, data_dir: str | Path, display_root: str | Path | None, incontext_data: bool) -> None:
    """Download and prepare DSBench data."""
    print(f"Preparing DSBench data for {split} split and saving to {data_dir}...")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_data_dir = data_dir / "data"

    # Extract if not already cached (hf_hub_download handles download caching)
    if not extracted_data_dir.exists():
        print("  Downloading dataset from HuggingFace...")
        zip_path = Path(
            hf_hub_download(repo_id="liqiang888/DSBench", filename="data_analysis/data.zip", repo_type="dataset")
        )
        print("    Extracting data...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        if not extracted_data_dir.exists():
            raise FileNotFoundError(f"Could not find data directory after extraction in {extracted_data_dir}")
        print(f"    Dataset cached to {data_dir}")
    else:
        print(f"  Using cached dataset from {data_dir}")

    # Load metadata
    print("  Loading metadata...")
    metadata_path = Path(
        hf_hub_download(repo_id="liqiang888/DSBench", filename="data_analysis/data.json", repo_type="dataset")
    )
    metadata = []
    with open(metadata_path, "r") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line.strip()))

    # Process all tasks
    if not display_root:
        display_root = extracted_data_dir
    else:
        display_root = Path(display_root)

    print(
        f"  Processing {len(metadata)} tasks at {extracted_data_dir} - using display root {display_root} for paths shown in the prompt..."
    )
    all_entries = []

    for task in metadata:
        task_id = task["id"]
        task_dir = extracted_data_dir / task_id

        if not task_dir.exists():
            raise FileNotFoundError(
                f"Task directory not found: {task_dir}. "
                f"Expected task {task_id} from metadata but directory is missing. "
                "Data extraction may have failed."
            )
        if len(task["answers"]) != len(task["questions"]):
            raise ValueError(
                f"Task {task_id}: mismatched questions ({len(task['questions'])}) "
                f"and answers ({len(task['answers'])}) counts in metadata."
            )

        # Read introduction
        intro_file = task_dir / "introduction.txt"
        introduction = ""
        if intro_file.exists():
            introduction = intro_file.read_text(encoding="utf-8", errors="ignore")

        # Get data files - support all Excel formats
        excel_files = []
        for ext in ["*.xlsx", "*.xlsb", "*.xlsm"]:
            excel_files.extend(task_dir.glob(ext))
        excel_files = [f for f in excel_files if "answer" not in f.name.lower()]

        # Read Excel content for in-context mode
        if incontext_data:
            excel_content = ""
            for excel_file in excel_files:
                sheets_text = read_excel_to_text(excel_file)
                excel_content += f"The excel file {excel_file.name} is: {sheets_text}\n\n"

        # Format paths for tool mode (relative to data directory)
        excel_paths = format_paths_for_prompt(excel_files, actual_root=extracted_data_dir, display_root=display_root)

        # Uncomment to get image files and csv files (for future multimodal and agentic support)
        # image_files = []
        # for ext in ["*.jpg", "*.png", "*.jpeg"]:
        #     image_files.extend(task_dir.glob(ext))
        # csv_files = list(task_dir.glob("*.csv"))

        # Process each question
        for idx, question_name in enumerate(task["questions"]):
            question_file = task_dir / f"{question_name}.txt"

            if not question_file.exists():
                print(f"    Warning: {task_id}/{question_name}.txt not found, skipping")
                continue

            question_text = question_file.read_text(encoding="utf-8", errors="ignore").strip()

            # Build problem text (introduction + question)
            problem_text = ""
            if introduction:
                problem_text += f"The introduction is detailed as follows.\n{introduction}\n\n"
            problem_text += f"The question for this task is detailed as follows.\n{question_text}"

            # Create entry with all necessary fields
            entry = {
                # Skills standard fields
                "problem": problem_text,
                "expected_answer": task["answers"][idx],
                # For tool mode
                "excel_paths": excel_paths,
                # Metadata
                "task_id": task_id,
                "question_id": question_name,
                "task_name": task["name"],
                "task_url": task["url"],
                "task_year": task["year"],
            }

            if incontext_data:
                entry["excel_content"] = excel_content.strip()

            all_entries.append(entry)

    # Validate we got some entries
    if not all_entries:
        raise ValueError(
            f"No valid entries created! Processed {len(metadata)} tasks but all failed. "
            "Check that data was downloaded correctly and Excel files are readable."
        )

    # Save to output file
    output_file = data_dir / f"{split}.jsonl"
    with open(output_file, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"  ✓ Saved {len(all_entries)} questions to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="DSBench only has test split")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Directory to save the data (defaults to dataset directory)"
    )
    parser.add_argument(
        "--display_root",
        type=str,
        default=None,
        help='Root directory to display in paths (absolute for abs paths, Path(".") for relative)',
    )
    parser.add_argument(
        "--incontext_data",
        action="store_true",
        help="Have the excel files read in-context under 'excel_content' field (Default: False)",
    )
    args = parser.parse_args()
    print(args)
    if args.data_dir is None:
        # Save to the same directory as this script
        data_dir = Path(__file__).absolute().parent
    else:
        data_dir = Path(args.data_dir)

    save_data(args.split, data_dir, args.display_root, args.incontext_data)
