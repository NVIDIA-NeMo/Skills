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

SUPPORTED_LANGUAGES = ["de", "es", "fr", "ja"]

# Regex to extract MCQ answer from \boxed{} output
EXTRACT_REGEX = r"\\boxed\{([A-D])\}"

# English instruction (from generic/general-boxed.yaml)
EN_INSTRUCTION = r"Solve the following problem. Make sure to put the answer (and only answer) inside \boxed{}."

# Language-specific instructions (from multilingual/general-boxed_{lang}.yaml)
BOXED_INSTRUCTIONS = {
    "de": r"Lösen Sie das folgende Problem. Stellen Sie sicher, dass Sie die Antwort (und nur die Antwort) in \boxed{} setzen.",
    "es": r"Resuelve el siguiente problema. Asegúrate de poner la respuesta (y solo la respuesta) dentro de \boxed{}.",
    "fr": r"Résolvez le problème suivant. Assurez-vous de mettre la réponse (et seulement la réponse) dans \boxed{}.",
    "ja": r"以下の問題を解いてください。答え（答えのみ）を\boxed{}の中に入れてください。",
}
