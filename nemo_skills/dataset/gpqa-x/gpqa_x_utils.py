# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# English few-shot template (from generic/general-boxed.yaml)
EN_FEW_SHOT_CONFIG = {
    "prefix": "Here are some examples of problems and solutions you can refer to.\n\n",
    "template": "Problem:\n{problem}\n\nSolution:\n{solution}\n\n\n\n\n\n",
    "suffix": "Here is the problem you need to solve:\n",
}

# Language-specific few-shot templates (from multilingual/general-boxed_{lang}.yaml)
FEW_SHOT_CONFIGS = {
    "de": {
        "prefix": "Hier sind einige Beispiele für Probleme und Lösungen, auf die Sie sich beziehen können.\n\n",
        "template": "Problem:\n{problem}\n\nLösung:\n{solution}\n\n\n\n\n\n",
        "suffix": "Hier ist das Problem, das Sie lösen müssen:\n",
    },
    "es": {
        "prefix": "Aquí tienes algunos ejemplos de problemas y soluciones que puedes consultar.\n\n",
        "template": "Problema:\n{problem}\n\nSolución:\n{solution}\n\n\n\n\n\n",
        "suffix": "Aquí tienes el problema que necesitas resolver:\n",
    },
    "fr": {
        "prefix": "Voici quelques exemples de problèmes et de solutions auxquels vous pouvez vous référer.\n\n",
        "template": "Problème:\n{problem}\n\nSolution:\n{solution}\n\n\n\n\n\n",
        "suffix": "Voici le problème que vous devez résoudre:\n",
    },
    "ja": {
        "prefix": "参考にできる問題と解答の例を以下に示します。\n\n",
        "template": "問題：\n{problem}\n\n解答：\n{solution}\n\n\n\n\n\n",
        "suffix": "解決すべき問題は以下の通りです：\n",
    },
}
