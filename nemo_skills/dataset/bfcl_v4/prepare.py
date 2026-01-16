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


import logging
import os

from nemo_skills.dataset.bfcl_v3.constants import (
    DATA_FOLDER_PATH,
)
from nemo_skills.dataset.bfcl_v3.prepare import download_and_process_bfcl_data
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# Github paths for BFCL
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"


def main():
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES

    LOG.warning(
        "Currently processing according to the OpenAI model style which works for most models, including Qwen/Nemotron/DeepSeek."
    )

    download_and_process_bfcl_data(
        REPO_URL,
        DATA_FOLDER_PATH,
        output_dir=os.path.join(os.path.dirname(__file__)),
        scoring_categories=ALL_SCORING_CATEGORIES,
    )


if __name__ == "__main__":
    main()
