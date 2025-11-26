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

from nemo_skills.inference.model import BaseModel

PROMPT_FORMAT = """[Instructions]
You are an expert mathematician. You are given a math problem and a proof. You need to extract the grading rubric from the proof.
The rubric should be a list of bullet points that the proof should follow to get full points.

[Format]
You must output the grading rubric in the following format:

<rubric>
...
</rubric>

[Example Problem]
Prove that the cubic equation $f(x)=x^3+x-1=0$ has exactly one real root, and show that this root lies in the interval $(0,1)$.

[Example Proof]
Consider $f(x)=x^3+x-1$. Note that $f$ is a polynomial, hence continuous on $R$. Compute $f(0)=-1$ and $f(1)=1$. Since $f$ is continuous and $f(0)$ and $f(1)$ have opposite signs, the Intermediate Value Theorem guarantees the existence of at least one $c\in(0,1)$ with $f(c)=0$.

Next, compute the derivative $f'(x)=3x^2+1$. For every real $x$, $3x^2\ge 0$, so $f'(x)\ge 1>0$. Thus $f'$ is strictly positive on $R$, which implies $f$ is strictly increasing on $R$. A strictly increasing continuous function can have at most one real root. Combining existence (from the IVT) and uniqueness (from strict monotonicity) shows that $f(x)=0$ has exactly one real root, and by the IVT step that root lies in $(0,1)$.

[Example Rubric]
<rubric>
- Defines the function $f(x)=x^3 + x - 1$ and notes continuity on $R$.
- Evaluates $f(0)$ and $f(1)$ to show a sign change across the interval $[0,1]$.
- Applies the Intermediate Value Theorem to conclude there exists at least one root in $(0,1)$.
- Computes $f'(x)$ and establishes that it is strictly positive for all real $x$.
- Deduces that $f(x)$ is strictly increasing and thus has at most one real root.
- Combines the existence (from IVT) and uniqueness (from monotonicity) arguments to conclude there is exactly one real root in $(0,1)$.
</rubric>

Now, extract the grading rubric from the following problem and proof:

[Problem]
{problem}

[Proof]
{proof}
"""


async def process_single(
    llm: BaseModel,
    datapoint: dict,
    llm_kwargs: dict,
) -> dict:
    problem, gt_proof = datapoint["problem"], datapoint["metadata"]["ground_truth_solution"]
    prompt = PROMPT_FORMAT.format(problem=problem, proof=gt_proof)
    for _ in range(10):
        result = await llm.generate_async(
            prompt=[{"role": "user", "content": prompt}],
            **llm_kwargs,
        )
        rubric = extract_rubric(result["generation"])
        if rubric:
            break
        else:
            print(f"Failed to extract rubric for {datapoint['problem']}, retrying...")
    return {
        **datapoint,
        "ground_truth_proof": gt_proof,
        "rubric": rubric,
    }


def extract_rubric(llm_response: str) -> str:
    if "<rubric>" not in llm_response or "</rubric>" not in llm_response:
        return ""
    return llm_response.split("<rubric>")[1].split("</rubric>")[0].strip()
