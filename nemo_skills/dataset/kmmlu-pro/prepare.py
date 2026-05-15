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
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

answer_map = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
}

# Matches the '정답: X' answer line the prompt instructs the model to produce.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-E])\)?\**"

# Surface the per-subject breakdown in English so non-Korean-speaking
# readers of the metrics output can interpret it. Names follow the
# subject's official English designation where one exists.
_SUBJECT_EN = {
    # Law
    "가족관계의등록등에관한법률": "Family Registration Act",
    "감정평가관계법규": "Appraisal Regulations",
    "공법": "Public Law",
    "공탁법": "Deposit Act",
    "관세법개론": "Introduction to Customs Law",
    "내국소비세법": "Domestic Consumption Tax Law",
    "노동법1": "Labor Law I",
    "노동법2": "Labor Law II",
    "민법": "Civil Law",
    "민법개론": "Introduction to Civil Law",
    "민사법": "Civil Procedure Law",
    "민사집행법": "Civil Execution Law",
    "보건의약관계법규": "Health and Pharmaceutical Regulations",
    "보험계약법": "Insurance Contract Law",
    "보험업법": "Insurance Business Law",
    "부동산등기법": "Real Estate Registration Law",
    "사회보험법": "Social Insurance Law",
    "산업재산권법": "Industrial Property Law",
    "상법": "Commercial Law",
    "상업등기법및비송사건절차법": "Commercial Registration and Non-Contentious Cases Procedure Law",
    "세법개론": "Introduction to Tax Law",
    "세법학개론": "Tax Law Studies",
    "행정소송법": "Administrative Litigation Law",
    "헌법": "Constitutional Law",
    "형사법": "Criminal Law",
    # Economics, Business, Real Estate
    "경영학": "Business Administration",
    "경영학개론": "Introduction to Business",
    "경제원론": "Economics",
    "경제학원론": "Principles of Economics",
    "무역영어": "Trade English",
    "부동산학원론": "Principles of Real Estate",
    "손해사정이론": "Loss Adjustment Theory",
    "재정학": "Public Finance",
    "회계학": "Accounting",
    "회계학개론": "Introduction to Accounting",
    # Medicine
    "구강악안면외과학": "Oral and Maxillofacial Surgery",
    "내과학1": "Internal Medicine I",
    "내과학2": "Internal Medicine II",
    "부인과학": "Gynecology",
    "소아과학": "Pediatrics",
    "신경정신과학": "Neuropsychiatry",
    "안이비인후과학": "Ophthalmology/Otolaryngology",
    "예방의학": "Preventive Medicine",
    "외과학": "Surgery",
    "의학각론": "Specialized Medicine",
    "의학총론": "General Medicine",
    # Dentistry
    "소아치과학/치과교정학": "Pediatric Dentistry/Orthodontics",
    "영상치의학/구강내과학/구강병리학": "Dental Radiology/Oral Medicine/Oral Pathology",
    "치과보존학": "Conservative Dentistry",
    "치과보철학": "Prosthodontics",
    "치과재료학/구강생물학": "Dental Materials/Oral Biology",
    "치주과학/구강보건학": "Periodontology/Oral Health",
    # Pharmacy and Korean medicine
    "본초학": "Materia Medica",
    "산업약학": "Industrial Pharmacy",
    "생명약학": "Life Pharmacy",
    "임상실무약학": "Clinical Pharmacy",
    "침구학": "Acupuncture and Moxibustion",
    "한방생리학": "Korean Medicine Physiology",
    "한약학 응용": "Applied Korean Pharmacology",
    "한약학기초": "Basic Korean Pharmacology",
    # Other
    "자연과학개론": "Introduction to Natural Sciences",
}


def format_entry(entry):
    return {
        "expected_answer": answer_map[entry["solution"]],
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subset_for_metrics": _SUBJECT_EN.get(entry["subject"], entry["subject"]),
        **get_mcq_fields(entry["question"], entry["options"]),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    dataset = load_dataset("LGAI-EXAONE/KMMLU-Pro")[args.split]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
