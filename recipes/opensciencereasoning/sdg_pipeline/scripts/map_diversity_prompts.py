#!/usr/bin/env python3
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

"""
Script to map random prompts and their answer regex patterns to dataset samples.
Each sample gets assigned a random prompt from the diversity prompts file.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List


def load_diversity_prompts(prompts_path: str, prompt_key: str = "prompt", answer_regex_key: str = "answer_regex") -> List[Dict]:
    """Load diversity prompts from JSONL file.
    
    Args:
        prompts_path: Path to the prompts file (JSON array or JSONL format)
        prompt_key: Field name for the prompt text
        answer_regex_key: Field name for the answer regex pattern
    
    Returns:
        List of dictionaries with 'prompt' and 'answer_regex' keys
    """
    prompts = []
    
    try:
        # Try to load as JSON array first
        with open(prompts_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content.startswith('['):
                # JSON array format
                data = json.loads(content)
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain a list of prompt objects")
                prompts = data
            else:
                # JSONL format - read line by line
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        prompt_obj = json.loads(line)
                        prompts.append(prompt_obj)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load prompts from {prompts_path}: {e}")
    
    # Extract prompts using specified field names
    normalized_prompts = []
    for i, prompt_obj in enumerate(prompts):
        if not isinstance(prompt_obj, dict):
            raise ValueError(f"Prompt {i} must be a dictionary")
        
        if prompt_key not in prompt_obj:
            raise ValueError(f"Prompt {i} missing required field '{prompt_key}'")
        if answer_regex_key not in prompt_obj:
            raise ValueError(f"Prompt {i} missing required field '{answer_regex_key}'")
            
        normalized_prompts.append({
            "prompt": prompt_obj[prompt_key],
            "answer_regex": prompt_obj[answer_regex_key]
        })
    
    return normalized_prompts


def map_prompts_to_dataset(input_file: str, output_file: str, prompts_path: str, prompt_key: str = "prompt", answer_regex_key: str = "answer_regex", seed: int = None):
    """Map random prompts to each sample in the dataset."""
    
    # Load diversity prompts
    diversity_prompts = load_diversity_prompts(prompts_path, prompt_key, answer_regex_key)
    print(f"Loaded {len(diversity_prompts)} diversity prompts")
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Process dataset
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Parse sample
            sample = json.loads(line)
            
            # Randomly select a prompt
            selected_prompt = random.choice(diversity_prompts)
            
            # Add prompt and answer_regex fields to the sample
            sample["prompt"] = selected_prompt["prompt"]
            sample["answer_regex"] = selected_prompt["answer_regex"]
            try:
                sample['_filled_prompt'] = selected_prompt["prompt"].format(**sample)
            except IndexError:
                logging.warning(
                    "Failed to fill prompt with sample fields; leaving _filled_prompt as original prompt: %s; sample keys:\n- %s",
                    selected_prompt["prompt"],
                    "\n- ".join(sample.keys())
                )
                sample['_filled_prompt'] = selected_prompt["prompt"]
                
            
            # Write enhanced sample
            outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')
            processed_count += 1
    
    print(f"Processed {processed_count} samples")
    print(f"Enhanced dataset saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Map random prompts and answer regex patterns to dataset samples"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input JSONL dataset file"
    )
    parser.add_argument(
        "output_file", 
        type=str,
        help="Output JSONL file with prompt and answer_regex fields added"
    )
    parser.add_argument(
        "--prompts_path",
        type=str,
        required=True,
        help="Path to JSON file containing diversity prompts"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="user_prompt",
        help="Field name for the prompt text in the prompts file"
    )
    parser.add_argument(
        "--answer_regex_key",
        type=str,
        default="output_regex",
        help="Field name for the answer regex pattern in the prompts file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible prompt assignment"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if not Path(args.prompts_path).exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_path}")
    
    # Create output directory if needed
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Map prompts to dataset
    map_prompts_to_dataset(
        input_file=args.input_file,
        output_file=args.output_file,
        prompts_path=args.prompts_path,
        prompt_key=args.prompt_key,
        answer_regex_key=args.answer_regex_key,
        seed=args.seed
    )


if __name__ == "__main__":
    main()