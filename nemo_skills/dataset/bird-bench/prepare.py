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

import argparse
import glob
import json
from pathlib import Path
import os
import re
import sqlite3
import wget
import zipfile

from datasets import load_dataset

def download_data(output_dir):
    #dataset = load_dataset("birdsql/bird_sql_dev_20251106", split="dev_20251106")

    # Download zip directly (HF Dataset is missing SQL files and table info)
    print("Downloading and extracting data file...")
    url = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
    filename = wget.download(url, out=output_dir)
    with zipfile.ZipFile(Path(output_dir, filename), 'r') as f_in:
        f_in.extractall(output_dir)

    # Expand tables zipfiles
    print("Extracting databases...")
    dev_dir = Path(output_dir, "dev_20240627/")
    dbs_zipfile = Path(dev_dir, "dev_databases.zip")
    with zipfile.ZipFile(dbs_zipfile, 'r') as f_dbs:
        f_dbs.extractall(dev_dir)

    print("Extracted all data!")
    return dev_dir


def read_tables_file(base_dir):
    """
    Gets each db's information by using sqlite3 to get a table dump.
    """
    tables_info = {}
    all_db_dirs = glob.glob("*", root_dir=os.path.join(base_dir, "dev_databases"))

    for db_dir in all_db_dirs:
        print(f"Reading database info from: {db_dir}")
        table_info = ""

        # Grab the db's sqlite file & read the dump
        full_db_dir = os.path.join(base_dir, "dev_databases", db_dir)
        sqlite_file = os.path.join(full_db_dir, db_dir + ".sqlite")
        assert os.path.exists(sqlite_file)

        con = sqlite3.connect(os.path.join(full_db_dir, db_dir + '.sqlite'))
        con.text_factory = lambda b: b.decode(errors = 'ignore')
        for line in con.iterdump():
            if line[:6] == "INSERT":
                line = line.replace('\n', ' ')
            line = re.sub(f" +", ' ', line)
            table_info += line + '\n'

        # Time to truncate any long INSERT chains (allow 10 max at once)
        insert_chain = r"((INSERT.*$\n){10})((INSERT.*\n)*)"
        table_info = re.sub(insert_chain, r"\1\n...\n", table_info, flags=re.MULTILINE)

        # Also get rid of any INSERT INTO * VALUES (...) <- lots of entries (>10)
        many_values = r"(?:VALUES )(((\([^)]*)\)[,;]\s*)){10}(.*)(?:;)"
        table_info = re.sub(many_values, r"...", table_info, flags=re.MULTILINE)

        tables_info[db_dir] = table_info

    return tables_info


def format_entries(file_path, tables_info, out_file):
    """
    Combines the raw BIRD data entries with corresponding table info and
    ground truth solution to form dev manifest
    """
    with open(out_file, 'w') as f_out:
        with open(file_path, 'r') as f_in:
            f_in.readline() # Discard first square bracket
            end_entry = r"  }"
            entry_str = ""

            for line in f_in:
                # Flatten & grab relevant key/values if end of entry reached
                entry_str += line.strip()
                if re.match(end_entry, line) is not None:
                    if entry_str[-1] == ',':
                        entry_str = entry_str[:-1]
                    entry = json.loads(entry_str)

                    final_entry = {
                        "question": entry["question"],
                        "solution": entry["SQL"],      #TODO: Check key for this
                        "sql_context": tables_info[entry["db_id"]]
                    }
                    f_out.write(json.dumps(final_entry))
                    f_out.write("\n")

                    entry_str = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    #dev_dir = download_data(args.output_dir)
    dev_dir = Path(args.output_dir, "dev_20240627/")

    print("Starting processing...")

    # First read tables data
    tables_info = read_tables_file(dev_dir)
    print("Finished reading tables.")

    format_entries(
        Path(dev_dir, "dev.json"),
        tables_info,
        Path(dev_dir, "dev_reformatted.json")
    )
    print("Finished formatting entries. All done!")



if __name__ == "__main__":
    main()
