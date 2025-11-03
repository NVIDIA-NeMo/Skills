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

"""Main entry point for model comparison analysis"""

import argparse
import logging
import os
from datetime import datetime

from .analyzer import OrganizedModelAnalyzer

logger = logging.getLogger(__name__)


def main():
    """Main function to run the organized analysis with UMAP"""

    parser = argparse.ArgumentParser(
        description="Compare language model outputs with comprehensive analysis including UMAP visualizations"
    )

    parser.add_argument(
        "--json_file", type=str, required=True, help="Input json file containing model outputs to compare"
    )

    parser.add_argument("--result_dir", type=str, required=True, help="Base directory for saving analysis results")

    args = parser.parse_args()

    # Extract dataset split from filename
    split = os.path.basename(args.json_file).split("_")[0]
    result_base_dir = f"{args.result_dir}/{split}"

    logger.info(f"Result base directory: {result_base_dir}")
    logger.info(f"Input JSON file: {args.json_file}")

    # Track total analysis time
    start = datetime.now()

    # Create analyzer instance
    analyzer = OrganizedModelAnalyzer(args.json_file, result_base_dir)

    # Load data and initialize models
    analyzer.load_data()
    analyzer.initialize_models()

    # Run comprehensive analysis with organized output and UMAP
    analyzer.generate_final_report()

    # Calculate and log total elapsed time
    end = datetime.now()
    elapsed = end - start
    elapsed_in_min = elapsed.total_seconds() / 60
    logger.info(f"Total elapsed time: {elapsed_in_min:.2f} minutes")


if __name__ == "__main__":
    main()
