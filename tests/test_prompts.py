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


from transformers import AutoTokenizer

from nemo_skills.prompt.utils import get_prompt, get_token_count


def test_get_token_count():
    tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)
    messages = [{"role": "user", "content": "hello"}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]

    assert get_token_count(tokenizer, "hello") == 1
    assert get_token_count(tokenizer, messages) == 17
    assert get_token_count(tokenizer, messages, tools=tools) == 266
    assert get_token_count(None, "hello") is None
    assert get_token_count(tokenizer, None) is None


def test_generic_math_problem_augmentation_prompt():
    prompt = get_prompt(
        "generic/problem-augmentation", "meta-llama/Llama-3.1-8B-Instruct", examples_type="math_problem_augmentation"
    )

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Write a new math problem inspired by a given one. Make the new problem reasonable and solvable.

Here are some examples of how to complete this task.

Original problem:
In the equation
$$5x^2-kx+1=0$$
determine $k$ such that the difference of the roots be equal to unity.

New problem:
The roots $x_1$ and $x_2$ of the equation
$$x^2-3ax+a^2=0$$
are such that
$x_1^2+x_2^2=1.75$.
Determine $a$.





Original problem:
Solve the following equation
$\\ds\\f{3+x}{3x}=\\sqrt {\\ds\\f{1}{9}+\\ds\\f{1}{x}\\sqrt {\\ds\\f{4}{9}+\\ds\\f{2}{x^2}}}$

New problem:
Solve the following equation
$\\sqrt {1+x\\sqrt {x^2+24}}=x+1$





Original problem:
In an infinitely decreasing geometric progression the sum
of all the terms occupying odd places is equal to 36, and that of all
the terms at even places equals 12.
Find the progression.

New problem:
The sum of the terms of an infinitely decreasing geometric
progression is equal to 56, and the sum of the squared terms of the
same progression is 448.
Find the first term and the common ratio.





Original problem:
Two railway stations are at a distance of 96 km from each other.
One train covers this distance 40 minutes faster than does the other.
The speed of the first train is 12 km/h higher than that of the second.
Determine the speed of both trains.

New problem:
A student was asked to multiply 78 by a two-digit number
in which the tens digit was three times as large as the units digit;
by mistake, he interchanged the digits in the second factor and
thus obtained a product smaller than the true product by 2808.
What was the true product?





Original problem:
What's the meaning of life?

Write another problem inspired by this one.
Don't just change the numbers and context, but try to create a problem that requires another approach to solve.
Start directly with the problem statement and DO NOT include any phrases such as "Here is a new math problem inspired by a given one".
After the problem is completed finish your response right away.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.fill({"problem": "What's the meaning of life?"}, format_as_string=True) == expected_prompt


def test_generic_codegen_prompt():
    prompt = get_prompt("generic/codegen")

    expected_prompt = [
        {
            "role": "user",
            "content": '''
Here is a problem for which you need to generate/complete code:
def 2_plus_2():
    """Write code to solve 2 + 2"""

Please continue to complete the function with python programming language. You are not allowed to modify the given code and do the completion only.

The solution should be in the following format:
```python
# Your code here
```'''.strip(),
        },
    ]
    assert prompt.fill({"question": 'def 2_plus_2():\n    """Write code to solve 2 + 2"""'}) == expected_prompt


def test_generic_default_prompt():
    prompt = get_prompt("generic/default")

    expected_prompt = [
        {
            "role": "user",
            "content": "How are you?",
        },
    ]
    assert prompt.fill({"question": "How are you?"}) == expected_prompt


def test_thinking_assistant_response():
    """Test that the thinking part is added to the assistant response."""
    prompt = get_prompt("generic/math", tokenizer="openai/gpt-oss-120b")

    expected_prompt = """<|channel|>analysis<|message|>Let me think step by step.... The answer is 4<|end|><|start|>assistant<|channel|>final<|message|>The answer is 4<|return|>"""
    output_prompt = prompt.format_assistant_response(
        content="The answer is 4", thinking="Let me think step by step.... The answer is 4"
    )

    assert output_prompt == expected_prompt, "Formatted assistant response is not as expected"


def test_generic_math_prompt():
    prompt = get_prompt("generic/math", "meta-llama/Llama-3.1-8B-Instruct")

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.

2 + 2 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.fill({"problem": "2 + 2 = ?"}, format_as_string=True) == expected_prompt


def test_generic_math_prompt_code_examples():
    prompt = get_prompt(
        "generic/math", tokenizer="meta-llama/Llama-3.1-8B", code_tags="nemotron", examples_type="math_text_with_code"
    )

    expected_prompt = """<|begin_of_text|>Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.

Here are some examples of problems and solutions you can refer to.

Problem:
A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?

Solution:
Let's write down an equation for the parabola and solve for $c$ using sympy.
<llm-code>
import sympy as sp

# define the symbols
x, y, b, c = sp.symbols('x y b c')

# define the parabola equation
parabola_eq = sp.Eq(y, x**2 + b*x + c)

# the parabola passes through the points (-1,-11) and (3,17)
# so we substitute these points into the parabola equation
point_1 = parabola_eq.subs({x: -1, y: -11})
point_2 = parabola_eq.subs({x: 3, y: 17})

# we now have two equations and two unknowns (b and c)
# we will solve for b and c
solutions = sp.solve((point_1,point_2), (b, c))
solutions[c]
</llm-code>
<llm-code-output>
-7
</llm-code-output>

So c is \\boxed{-7}





Problem:
Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?

Enter "odd", "even", or "neither".

Solution:
The definition of an odd function is that $f(-x) = -f(x)$.
Applying this to $f(f(-x))$ we get $f(f(-x)) = f(-f(x)) = -f(f(x))$.
Thus, $f(f(x))$ is an \\boxed{odd} function.





Problem:
At the 2007 Math Olympics, Team Canada won $17$ out of a possible $100$ medals. Which one of the following is closest to the fraction of medals that they won? $$
\\frac{1}{4} \\qquad \\frac{1}{5} \\qquad \\frac{1}{6} \\qquad \\frac{1}{7} \\qquad \\frac{1}{8}
$$

Solution:
Let's use sympy to print out the differences between the fraction of medals that Canada won and each of the options.
<llm-code>
from sympy import Rational, Abs

# team Canada won 17 out of 100 medals
medal_frac = Rational(17, 100)

# list of options
options = [Rational(1, 4), Rational(1, 5), Rational(1, 6), Rational(1, 7), Rational(1, 8)]

# let's print out the differences
[Abs(medal_frac - frac_option) for frac_option in options]
</llm-code>
<llm-code-output>
[2/25, 3/100, 1/300, 19/700, 9/200]
</llm-code-output>

Let's now check which difference is the smallest.
<llm-code>
import numpy as np

# Calculate the idx of the closest option
min_idx = np.argmin([2/25, 3/100, 1/300, 19/700, 9/200])

# Print the closest option
print(options[min_idx])
</llm-code>
<llm-code-output>
1/6
</llm-code-output>

So the answer is \\boxed{1/6}.





Problem:
A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?

Solution:
Let the dimensions of $P$ be $x$, $y$, and $z$.
The sum of the box's edges is $4(x + y + z)$ and the surface area is $2xy + 2yz + 2xz$.
The diameter of the sphere is $2r$ and it's equal to the diagonal of the box.
Let's now write down the equations based on the above information and solve them using sympy.
<llm-code>
from sympy import symbols, Eq, solve

# define the variables
x, y, z, r = symbols('x y z r')

# equations based on the given information
eq1 = Eq(2 * (x*y + x*z + y*z), 384)
eq2 = Eq(4 * (x + y + z), 112)

# the diagonal of the box is the diameter of the sphere
diagonal_eq = Eq(x**2 + y**2 + z**2, (2*r)**2)

# solve the equations
solutions = solve((eq1, eq2, diagonal_eq), (x, y, z, r))

# let's see which values of r we get
[solution[-1] for solution in solutions]
</llm-code>
<llm-code-output>
[-10, -10, 10, 10]
</llm-code-output>

Since the radius of the sphere is positive, we get $r = \\boxed{10}$.





Problem:
A bee starts flying from point $P_0$. She flies $1$ inch due east to point $P_1$. For $j \\ge 1$, once the bee reaches point $P_j$, she turns $30^{\\circ}$ counterclockwise and then flies $j+1$ inches straight to point $P_{j+1}$. When the bee reaches $P_{2015},$ how far from $P_0$ is she, in inches?

Solution:
We can represent the rotation action via $\\omega = e^{\\pi i/6}$.
Let's assume the bee starts at the origin, so $P_{2015}$ is at the point \\[z = 1 + 2 \\omega + 3 \\omega^2 + 4 \\omega^3 + \\dots + 2015 \\omega^{2014}.\\]
This is an arithmetic-geometric series which we can solve by simplifying the expression.
Alternatively, we can solve for |z| using sympy using the following code.
<llm-code>
from sympy import I, pi, exp, sqrt

# rotation action of 30 degrees
omega = exp(I * pi/6)

position = 0

for i in range(2015):
    delta = (i + 1) * omega**(i)
    position += delta

real, img = (position.as_real_imag())
# Distance from origin i.e. starting point
dist = sqrt(real**2 + img**2)
print(dist)
</llm-code>
<llm-code-output>
sqrt(2)*(1008 + 1008*sqrt(3))
</llm-code-output>

So the bee is $\\boxed{1008\\sqrt{2} + 1008\\sqrt{6}}$ far from the starting point.





Here is the problem you need to solve:
2 + 2 = ?"""
    assert prompt.fill({"problem": "2 + 2 = ?"}, format_as_string=True) == expected_prompt


def test_llama_code_output_format_examples():
    prompt = get_prompt(
        "generic/math", "meta-llama/Llama-3.1-8B-Instruct", "llama3", examples_type="math_text_with_code"
    )

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.

Here are some examples of problems and solutions you can refer to.

Problem:
A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?

Solution:
Let's write down an equation for the parabola and solve for $c$ using sympy.
<|python_tag|>import sympy as sp

# define the symbols
x, y, b, c = sp.symbols('x y b c')

# define the parabola equation
parabola_eq = sp.Eq(y, x**2 + b*x + c)

# the parabola passes through the points (-1,-11) and (3,17)
# so we substitute these points into the parabola equation
point_1 = parabola_eq.subs({x: -1, y: -11})
point_2 = parabola_eq.subs({x: 3, y: 17})

# we now have two equations and two unknowns (b and c)
# we will solve for b and c
solutions = sp.solve((point_1,point_2), (b, c))
solutions[c]
<|eom_id|><|start_header_id|>ipython<|end_header_id|>

completed
[stdout]
-7
[/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>


So c is \\boxed{-7}





Problem:
Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?

Enter "odd", "even", or "neither".

Solution:
The definition of an odd function is that $f(-x) = -f(x)$.
Applying this to $f(f(-x))$ we get $f(f(-x)) = f(-f(x)) = -f(f(x))$.
Thus, $f(f(x))$ is an \\boxed{odd} function.





Problem:
At the 2007 Math Olympics, Team Canada won $17$ out of a possible $100$ medals. Which one of the following is closest to the fraction of medals that they won? $$
\\frac{1}{4} \\qquad \\frac{1}{5} \\qquad \\frac{1}{6} \\qquad \\frac{1}{7} \\qquad \\frac{1}{8}
$$

Solution:
Let's use sympy to print out the differences between the fraction of medals that Canada won and each of the options.
<|python_tag|>from sympy import Rational, Abs

# team Canada won 17 out of 100 medals
medal_frac = Rational(17, 100)

# list of options
options = [Rational(1, 4), Rational(1, 5), Rational(1, 6), Rational(1, 7), Rational(1, 8)]

# let's print out the differences
[Abs(medal_frac - frac_option) for frac_option in options]
<|eom_id|><|start_header_id|>ipython<|end_header_id|>

completed
[stdout]
[2/25, 3/100, 1/300, 19/700, 9/200]
[/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>


Let's now check which difference is the smallest.
<|python_tag|>import numpy as np

# Calculate the idx of the closest option
min_idx = np.argmin([2/25, 3/100, 1/300, 19/700, 9/200])

# Print the closest option
print(options[min_idx])
<|eom_id|><|start_header_id|>ipython<|end_header_id|>

completed
[stdout]
1/6
[/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>


So the answer is \\boxed{1/6}.





Problem:
A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?

Solution:
Let the dimensions of $P$ be $x$, $y$, and $z$.
The sum of the box's edges is $4(x + y + z)$ and the surface area is $2xy + 2yz + 2xz$.
The diameter of the sphere is $2r$ and it's equal to the diagonal of the box.
Let's now write down the equations based on the above information and solve them using sympy.
<|python_tag|>from sympy import symbols, Eq, solve

# define the variables
x, y, z, r = symbols('x y z r')

# equations based on the given information
eq1 = Eq(2 * (x*y + x*z + y*z), 384)
eq2 = Eq(4 * (x + y + z), 112)

# the diagonal of the box is the diameter of the sphere
diagonal_eq = Eq(x**2 + y**2 + z**2, (2*r)**2)

# solve the equations
solutions = solve((eq1, eq2, diagonal_eq), (x, y, z, r))

# let's see which values of r we get
[solution[-1] for solution in solutions]
<|eom_id|><|start_header_id|>ipython<|end_header_id|>

completed
[stdout]
[-10, -10, 10, 10]
[/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>


Since the radius of the sphere is positive, we get $r = \\boxed{10}$.





Problem:
A bee starts flying from point $P_0$. She flies $1$ inch due east to point $P_1$. For $j \\ge 1$, once the bee reaches point $P_j$, she turns $30^{\\circ}$ counterclockwise and then flies $j+1$ inches straight to point $P_{j+1}$. When the bee reaches $P_{2015},$ how far from $P_0$ is she, in inches?

Solution:
We can represent the rotation action via $\\omega = e^{\\pi i/6}$.
Let's assume the bee starts at the origin, so $P_{2015}$ is at the point \\[z = 1 + 2 \\omega + 3 \\omega^2 + 4 \\omega^3 + \\dots + 2015 \\omega^{2014}.\\]
This is an arithmetic-geometric series which we can solve by simplifying the expression.
Alternatively, we can solve for |z| using sympy using the following code.
<|python_tag|>from sympy import I, pi, exp, sqrt

# rotation action of 30 degrees
omega = exp(I * pi/6)

position = 0

for i in range(2015):
    delta = (i + 1) * omega**(i)
    position += delta

real, img = (position.as_real_imag())
# Distance from origin i.e. starting point
dist = sqrt(real**2 + img**2)
print(dist)
<|eom_id|><|start_header_id|>ipython<|end_header_id|>

completed
[stdout]
sqrt(2)*(1008 + 1008*sqrt(3))
[/stdout]<|eot_id|><|start_header_id|>assistant<|end_header_id|>


So the bee is $\\boxed{1008\\sqrt{2} + 1008\\sqrt{6}}$ far from the starting point.





Here is the problem you need to solve:
2 + 2 = ?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert prompt.fill({"problem": "2 + 2 = ?"}, format_as_string=True) == expected_prompt


def test_qwen_code_output_format_examples():
    prompt = get_prompt(
        "generic/math", "Qwen/Qwen2.5-32B-Instruct", "qwen", examples_type="math_text_with_code", system_message=""
    )

    expected_prompt = """<|im_start|>system
<|im_end|>
<|im_start|>user
Solve the following math problem. Make sure to put the answer (and only answer) inside \\boxed{}.

Here are some examples of problems and solutions you can refer to.

Problem:
A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?

Solution:
Let's write down an equation for the parabola and solve for $c$ using sympy.
```python
import sympy as sp

# define the symbols
x, y, b, c = sp.symbols('x y b c')

# define the parabola equation
parabola_eq = sp.Eq(y, x**2 + b*x + c)

# the parabola passes through the points (-1,-11) and (3,17)
# so we substitute these points into the parabola equation
point_1 = parabola_eq.subs({x: -1, y: -11})
point_2 = parabola_eq.subs({x: 3, y: 17})

# we now have two equations and two unknowns (b and c)
# we will solve for b and c
solutions = sp.solve((point_1,point_2), (b, c))
solutions[c]
```
```output
-7
```

So c is \\boxed{-7}





Problem:
Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?

Enter "odd", "even", or "neither".

Solution:
The definition of an odd function is that $f(-x) = -f(x)$.
Applying this to $f(f(-x))$ we get $f(f(-x)) = f(-f(x)) = -f(f(x))$.
Thus, $f(f(x))$ is an \\boxed{odd} function.





Problem:
At the 2007 Math Olympics, Team Canada won $17$ out of a possible $100$ medals. Which one of the following is closest to the fraction of medals that they won? $$
\\frac{1}{4} \\qquad \\frac{1}{5} \\qquad \\frac{1}{6} \\qquad \\frac{1}{7} \\qquad \\frac{1}{8}
$$

Solution:
Let's use sympy to print out the differences between the fraction of medals that Canada won and each of the options.
```python
from sympy import Rational, Abs

# team Canada won 17 out of 100 medals
medal_frac = Rational(17, 100)

# list of options
options = [Rational(1, 4), Rational(1, 5), Rational(1, 6), Rational(1, 7), Rational(1, 8)]

# let's print out the differences
[Abs(medal_frac - frac_option) for frac_option in options]
```
```output
[2/25, 3/100, 1/300, 19/700, 9/200]
```

Let's now check which difference is the smallest.
```python
import numpy as np

# Calculate the idx of the closest option
min_idx = np.argmin([2/25, 3/100, 1/300, 19/700, 9/200])

# Print the closest option
print(options[min_idx])
```
```output
1/6
```

So the answer is \\boxed{1/6}.





Problem:
A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?

Solution:
Let the dimensions of $P$ be $x$, $y$, and $z$.
The sum of the box's edges is $4(x + y + z)$ and the surface area is $2xy + 2yz + 2xz$.
The diameter of the sphere is $2r$ and it's equal to the diagonal of the box.
Let's now write down the equations based on the above information and solve them using sympy.
```python
from sympy import symbols, Eq, solve

# define the variables
x, y, z, r = symbols('x y z r')

# equations based on the given information
eq1 = Eq(2 * (x*y + x*z + y*z), 384)
eq2 = Eq(4 * (x + y + z), 112)

# the diagonal of the box is the diameter of the sphere
diagonal_eq = Eq(x**2 + y**2 + z**2, (2*r)**2)

# solve the equations
solutions = solve((eq1, eq2, diagonal_eq), (x, y, z, r))

# let's see which values of r we get
[solution[-1] for solution in solutions]
```
```output
[-10, -10, 10, 10]
```

Since the radius of the sphere is positive, we get $r = \\boxed{10}$.





Problem:
A bee starts flying from point $P_0$. She flies $1$ inch due east to point $P_1$. For $j \\ge 1$, once the bee reaches point $P_j$, she turns $30^{\\circ}$ counterclockwise and then flies $j+1$ inches straight to point $P_{j+1}$. When the bee reaches $P_{2015},$ how far from $P_0$ is she, in inches?

Solution:
We can represent the rotation action via $\\omega = e^{\\pi i/6}$.
Let's assume the bee starts at the origin, so $P_{2015}$ is at the point \\[z = 1 + 2 \\omega + 3 \\omega^2 + 4 \\omega^3 + \\dots + 2015 \\omega^{2014}.\\]
This is an arithmetic-geometric series which we can solve by simplifying the expression.
Alternatively, we can solve for |z| using sympy using the following code.
```python
from sympy import I, pi, exp, sqrt

# rotation action of 30 degrees
omega = exp(I * pi/6)

position = 0

for i in range(2015):
    delta = (i + 1) * omega**(i)
    position += delta

real, img = (position.as_real_imag())
# Distance from origin i.e. starting point
dist = sqrt(real**2 + img**2)
print(dist)
```
```output
sqrt(2)*(1008 + 1008*sqrt(3))
```

So the bee is $\\boxed{1008\\sqrt{2} + 1008\\sqrt{6}}$ far from the starting point.





Here is the problem you need to solve:
2 + 2 = ?<|im_end|>
<|im_start|>assistant
"""
    assert prompt.fill({"problem": "2 + 2 = ?"}, format_as_string=True) == expected_prompt


def test_judge_arena():
    prompt = get_prompt("judge/arena")

    expected_prompt = [
        {
            "role": "system",
            "content": """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]".
""".strip(),
        },
        {
            "role": "user",
            "content": """
<|User Prompt|>
What's better for a cold: tea or coffee?

<|The Start of Assistant A's Answer|>
Tea for sure
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
I mean, coffee, why do you even ask?
<|The End of Assistant B's Answer|>
""".strip(),
        },
    ]
    assert (
        prompt.fill(
            {
                "question": "What's better for a cold: tea or coffee?",
                "answer_1": "Tea for sure",
                "answer_2": "I mean, coffee, why do you even ask?",
            }
        )
        == expected_prompt
    )


def test_judge_math():
    prompt = get_prompt("judge/math")

    expected_prompt = [
        {
            "role": "user",
            "content": """
You will be asked to look at the two answers (predicted and expected) to a math problem and to judge whether they are equivalent within the context of the problem.

Please first explain your reasoning in a couple of sentences. Then respond with only Yes or No as your judgement on whether the two answers are the same.
When comparing answers only perform trivial simplifications.

Here are a few examples. Please include both "Reasoning" and "Judgement" in your final response in the same format as below.


Example 1:
Problem: Factor $7x^3 - 21x^2 + 14x$.
Predicted answer: $7x(x - 2)(x - 1)$
Expected answer: $7x(x-1)(x-2)$

Reasoning: The order of the factors does not matter, so the answers are the same.
Judgement: Yes


Example 2:
Problem: A rectangle has a length of 6 meters and a width of 2 meters. If the length is reduced by 3 meters and the width is halved, what is the new area of the rectangle in square meters?
Predicted answer: 3/2
Expected answer: 1.5

Reasoning: 3/2 is the same as 1.5
Judgement: Yes


Example 3:
Problem: Simplify the expression $\\sqrt{7!}$, where $n!$ stands for $n\\cdot(n-1)\\cdot(n-2)\\cdots \\cdot 2\\cdot 1$.
Predicted answer: 71
Expected answer: 12\\sqrt{35}.

Reasoning: This is non-trivial to simplify, so the answers are different.
Judgement: No


Example 4:
Problem: What is the simplified form of the expression $\\sqrt{98 x^{3} y^{5} z} ?
\\begin{align*}
\\text{A)} & 2 x y z \\sqrt{7 x y z} &
\\text{B)} &  7 x^{2} y^{2} \\sqrt{2 y z}
\\\\
\\text{C)} & 7 x y^{2} \\sqrt{2 x y z}  &
\\text{D)} &49 x y^{2} \\sqrt{2 x y z}
\\\\
\\end{align*}
Predicted answer: 7 x y^{2} \\\\sqrt{2 x y z}
Expected answer: C

Reasoning: Predicted answer is the same as the expected answer choice C.
Judgement: Yes


Example 5:
Problem: A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint at $(4, b)$. Find all possible values of $b$, separated by commas.
Predicted answer:  -2, 6
Expected answer: 6, -2

Reasoning: The order doesn't matter in the context of the problem.
Judgement: Yes


Example 6:
Problem: Solve $\\tan x = \\sin x$ for $0 \\le x \\le 2 \\pi.$  Enter all the solutions, separated by commas.
Predicted answer: 0, \\pi
Expected answer: 0,\\pi,2\\pi.

Reasoning: Number of solutions is different.
Judgement: No


YOUR TASK

Problem: 1 + 1
Predicted answer: eh, 15?
Expected answer: 2
""".strip(),
        },
    ]
    assert (
        prompt.fill(
            {
                "problem": "1 + 1",
                "predicted_answer": "eh, 15?",
                "expected_answer": "2",
            }
        )
        == expected_prompt
    )


def test_judge_check_contamination():
    prompt = get_prompt("judge/check-contamination")

    expected_prompt = [
        {
            "role": "user",
            "content": """
Help me determine if the following two math problems are the same.

First problem: 1 + 3
Second problem: what's 3 plus 1?

Disregard the names and minor changes in word order that appear within.
If the two problems are very similar and if they produce the same answer, we consider them to be the same problem.
Respond with only "True" (problems are the same) or "False" (problems are different). Do not respond with anything else.
""".strip(),
        },
    ]
    assert (
        prompt.fill(
            {
                "problem1": "1 + 3",
                "problem2": "what's 3 plus 1?",
            }
        )
        == expected_prompt
    )


def test_generic_general_boxed_prompt():
    prompt = get_prompt("generic/general-boxed", "meta-llama/Llama-3.1-8B-Instruct")

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}.

What is the square root of 81 squared?
A. 9^2
B. 27
C. 81^2
D. 729
E. 6561
F. 12
G. 162
H. 243
I. 81
J. 9<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert (
        prompt.fill(
            {
                "problem": "What is the square root of 81 squared?\nA. 9^2\nB. 27\nC. 81^2\nD. 729\nE. 6561\nF. 12\nG. 162\nH. 243\nI. 81\nJ. 9"
            },
            format_as_string=True,
        )
        == expected_prompt
    )


def test_llm_as_judge_hle_original_prompt():
    prompt = get_prompt("judge/hle", "meta-llama/Llama-3.1-8B-Instruct")

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: If $\\sqrt{5+n}=7$, then what is the value of $n$?

[response]: Let's solve the problem step by step. 5+n=49, so n=44.

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: 44

Reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

Judgement: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

Confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    assert (
        prompt.fill(
            {
                "problem": "If $\\sqrt{5+n}=7$, then what is the value of $n$?",
                "generation": "Let's solve the problem step by step. 5+n=49, so n=44.",
                "expected_answer": "44",
            },
            format_as_string=True,
        )
        == expected_prompt
    )
