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

few_shots_topics = {
    "topic": {
        "Chemistry": "Calculate the molar mass of a sample of water (H₂O) given that the atomic mass of Hydrogen is 1.008 amu and the atomic mass of Oxygen is 16.00 amu.",
        "Other": "What is the capital of France?",
    },
    "subtopic": {
        "Chemistry": {
            "Organic Chemistry": "Draw the structural formula of ethanol (C_{2}H_{5}OH) and identify the functional group present.",
            "Other": "What is the capital of France?",
        },
        "Physics": {
            "Quantum Mechanics": "What is the Heisenberg Uncertainty Principle, and how does it limit the simultaneous measurement of an electron's position and momentum?",
            "Other": "What is the capital of France?",
        },
        "Biology": {
            "Genetics": "If one parent has the genotype Aa and the other has genotype aa, what is the probability that their offspring will have the homozygous recessive genotype?",
            "Other": "What is the capital of France?",
        },
    },
}
