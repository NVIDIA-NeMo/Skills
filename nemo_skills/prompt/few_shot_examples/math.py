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


# Source https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math/utils.py#L42
math_standard_few_shot = [
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.",
    },
    {
        "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
        "solution": "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$",
    },
    {
        "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "solution": "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}",
    },
    {
        "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "solution": "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$",
    },
]


math_text_detailed = [
    {
        "problem": "A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?",
        "solution": """
From the question we know that points $(-1, -11)$ and $(3, 17)$ lie on the parabola. This means that when we substitute $x$ and $y$ from these points into the equation $y = x^2 + bx + c$, the equation must hold true. We substitute these two points into the given equation to solve for $c$.

For the point $(-1, -11)$:

Substitute $x = -1$ and $ y = -11 $ into the equation:
\\[ -11 = (-1)^2 + b(-1) + c \\Rightarrow -11 = 1 - b + c \\Rightarrow -b + c = -12 \\]

For the point $(3, 17)$:

Substitute $x = 3$ and $y = 17$ into the equation:
\\[ 17 = (3)^2 + b(3) + c \\Rightarrow 17 = 9 + 3b + c \\Rightarrow 3b + c = 8 \\]

In summary, we have the two equations
\\begin{align*}
-b + c &= -12\\\\
3b + c &= 8
\\end{align*}

To solve for $c$ we can eliminate $b$ by multiplying the first equation by 3 and adding equations together.
Multiplying the first equation by 3, we have $3(-b + c) = 3 (-12) \\Rightarrow -3b + 3c = -36$. Adding equations together gives us
\\[ (-3b + 3c) + (3b + c) = -36 + 8 \\Rightarrow -3b + 3b + 3c + c = -28 \\Rightarrow 4c = -28 \\Rightarrow c = -28 : 4 \\Rightarrow c = \\boxed{-7} \\]
""".strip(),
    },
    {
        "problem": 'Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?\n\nEnter "odd", "even", or "neither".',
        "solution": """
To determine whether $f(f(x))$ is even, odd, or neither, we need to use the property of $f(x)$ being an odd function.

An odd function is defined as:
\\[ f(-x) = -f(x) \\quad \\text{for all } x \\]

Given that $f(x)$ is odd, let's find $f(f(-x))$ and see how it relates to $f(f(x))$.

1. Substitute $-x$ into the function $f(x)$:
\\[ f(-x) \\]

2. Since $f(x)$ is odd, apply the definition of an odd function:
\\[ f(-x) = -f(x) \\]

3. Now substitute $-f(x)$ into the function $f$:
\\[ f(f(-x)) = f(-f(x)) \\]

4. Again, using the fact that $f(x)$ is odd, apply the definition:
\\[ f(-f(x)) = -f(f(x)) \\]

5. We have found that:
\\[ f(f(-x)) = -f(f(x)) \\]

This matches the definition of an odd function.

So, the answer is:
\\[ \\boxed{\\text{odd}} \\]
""".strip(),
    },
    {
        "problem": "A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?",
        "solution": """
Let the dimensions of the rectangular box $P$ be $x$, $y$, and $z$. We know the following:

1. The sum of the lengths of the edges of $P$ is
\\[ 4(x + y + z) = 112 \\Rightarrow x + y + z = 112 : 4 \\Rightarrow x + y + z = 28 \\]

2. The surface area of $P$ is
\\[ 2xy + 2yz + 2xz = 384 \\Rightarrow xy + yz + xz = 384 : 2 \\Rightarrow xy + yz + xz = 192 \\]

Since the box is inscribed in the sphere, the diagonal of the box is the diameter of the sphere. The length of the diagonal is $\\sqrt{x^2 + y^2 + z^2}$

The diameter of the sphere is $2r$, so:
\\[ 2r = \\sqrt{x^2 + y^2 + z^2} \\Rightarrow (2r)^2 = x^2 + y^2 + z^2 = (x + y + z)^2 - (2xy + 2yz + 2xz) \\]

Substitute the known values:
\\[ 4r^2 = 28^2 - 384 = 784 - 384 = 400 \\Rightarrow r^2 = 100 \\Rightarrow r = \\boxed{10} \\]
""".strip(),
    },
    {
        "problem": "Let $\\mathbf{a} = \\begin{pmatrix} 2 \\\\ 1 \\\\ 5 \\end{pmatrix}.$  Find the vector $\\mathbf{b}$ such that $\\mathbf{a} \\cdot \\mathbf{b} = 11$ and\n\\[\\mathbf{a} \\times \\mathbf{b} = \\begin{pmatrix} -13 \\\\ -9 \\\\ 7 \\end{pmatrix}.\\]",
        "solution": """
Let $\\mathbf{b} = \\begin{pmatrix} x \\\\ y \\\\ z \\end{pmatrix}$.

First, use the dot product condition:
\\[ \\mathbf{a} \\cdot \\mathbf{b} = 11 \\Rightarrow 2x + y + 5z = 11 \\]

Next, use the cross product condition:
\\[ \\mathbf{a} \\times \\mathbf{b} = \\begin{pmatrix} 2 \\\\ 1 \\\\ 5 \\end{pmatrix} \\times \\begin{pmatrix} x \\\\ y \\\\ z \\end{pmatrix} = \\begin{pmatrix} -5y + z \\\\ 5x - 2z \\\\ -x + 2y \\end{pmatrix} = \\begin{pmatrix} -13 \\\\ -9 \\\\ 7 \\end{pmatrix} \\]

This gives us the system of equations:
   \\begin{align*}
   2x + y + 5z = 11 \\quad &(1) \\\\
   -5y + z = -13 \\quad &(2) \\\\
   5x - 2z = -9 \\quad &(3) \\\\
   -x + 2y = 7 \\quad &(4)
   \\end{align*}

Solve for $x$, $y$, and $z$ step-by-step:

From (2), $z = 5y - 13$.
From (4), $x = 2y - 7$.

Substitute $z = 5y - 13$ into (1):
\\[ 2(2y - 7) + y + 5(5y - 13) = 11 \\Rightarrow 4y - 14 + y + 25y - 65 = 11 \\Rightarrow 30y - 79 = 11 \\Rightarrow 30y = 90 \\Rightarrow y = 3 \\]

Now find $x$ and $z$:
\\[ x = 2y - 7 = 2(3) - 7 = -1 \\]

\\[ z = 5y - 13 = 5(3) - 13 = 2 \\]

Thus, the vector $\\mathbf{b}$ is:
\\[ \\mathbf{b} = \\boxed{\\begin{pmatrix} -1 \\\\ 3 \\\\ 2 \\end{pmatrix}} \\]
""".strip(),
    },
]

math_text_with_code = [
    {
        "problem": "A parabola with equation $y=x^2+bx+c$ passes through the points $(-1,-11)$ and $(3,17)$. What is $c$?",
        "solution": """
Let's write down an equation for the parabola and solve for $c$ using sympy.
{code_begin}import sympy as sp

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
{code_end}{code_output_begin}
-7
{code_output_end}
So c is \\boxed{-7}
""".strip(),
    },
    {
        "problem": 'Let $f(x)$ be an odd function.  Is $f(f(x))$ even, odd, or neither?\n\nEnter "odd", "even", or "neither".',
        "solution": """
The definition of an odd function is that $f(-x) = -f(x)$.
Applying this to $f(f(-x))$ we get $f(f(-x)) = f(-f(x)) = -f(f(x))$.
Thus, $f(f(x))$ is an \\boxed{odd} function.
""".strip(),
    },
    {
        "problem": "At the 2007 Math Olympics, Team Canada won $17$ out of a possible $100$ medals. Which one of the following is closest to the fraction of medals that they won? $$\n\\frac{1}{4} \\qquad \\frac{1}{5} \\qquad \\frac{1}{6} \\qquad \\frac{1}{7} \\qquad \\frac{1}{8}\n$$",
        "solution": """
Let's use sympy to print out the differences between the fraction of medals that Canada won and each of the options.
{code_begin}from sympy import Rational, Abs

# team Canada won 17 out of 100 medals
medal_frac = Rational(17, 100)

# list of options
options = [Rational(1, 4), Rational(1, 5), Rational(1, 6), Rational(1, 7), Rational(1, 8)]

# let's print out the differences
[Abs(medal_frac - frac_option) for frac_option in options]
{code_end}{code_output_begin}
[2/25, 3/100, 1/300, 19/700, 9/200]
{code_output_end}
Let's now check which difference is the smallest.
{code_begin}import numpy as np

# Calculate the idx of the closest option
min_idx = np.argmin([2/25, 3/100, 1/300, 19/700, 9/200])

# Print the closest option
print(options[min_idx])
{code_end}{code_output_begin}
1/6
{code_output_end}
So the answer is \\boxed{1/6}.
""".strip(),
    },
    {
        "problem": "A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?",
        "solution": """
Let the dimensions of $P$ be $x$, $y$, and $z$.
The sum of the box's edges is $4(x + y + z)$ and the surface area is $2xy + 2yz + 2xz$.
The diameter of the sphere is $2r$ and it's equal to the diagonal of the box.
Let's now write down the equations based on the above information and solve them using sympy.
{code_begin}from sympy import symbols, Eq, solve

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
{code_end}{code_output_begin}
[-10, -10, 10, 10]
{code_output_end}
Since the radius of the sphere is positive, we get $r = \\boxed{10}$.
""".strip(),
    },
    {
        "problem": "A bee starts flying from point $P_0$. She flies $1$ inch due east to point $P_1$. For $j \\ge 1$, once the bee reaches point $P_j$, she turns $30^{\\circ}$ counterclockwise and then flies $j+1$ inches straight to point $P_{j+1}$. When the bee reaches $P_{2015},$ how far from $P_0$ is she, in inches?",
        "solution": """
We can represent the rotation action via $\\omega = e^{\\pi i/6}$.
Let's assume the bee starts at the origin, so $P_{2015}$ is at the point \\[z = 1 + 2 \\omega + 3 \\omega^2 + 4 \\omega^3 + \\dots + 2015 \\omega^{2014}.\\]
This is an arithmetic-geometric series which we can solve by simplifying the expression.
Alternatively, we can solve for |z| using sympy using the following code.
{code_begin}from sympy import I, pi, exp, sqrt

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
{code_end}{code_output_begin}
sqrt(2)*(1008 + 1008*sqrt(3))
{code_output_end}
So the bee is $\\boxed{1008\\sqrt{2} + 1008\\sqrt{6}}$ far from the starting point.
""".strip(),
    },
]


math_text_with_code_algebra = [
    {
        "problem": "John is 24 years younger than his dad. The sum of their ages is 68 years. How many years old is John?",
        "solution": """
To compute John's age, we can use Python's sympy library.
{code_begin}import sympy as sp

# Let's denote John's age by j
j = sp.symbols('j')

# Let's denote John's dad's age by d. John is 24 years younger than his dad
d = j + 24

# Sum of their ages is 68
eq = sp.Eq(j + d, 68)

# solving for j we get the John's age
sp.solve(eq, j)
{code_end}{code_output_begin}
[22]
{code_output_end}
So John is \\boxed{22} years old.
""".strip(),
    },
    {
        "problem": "Simplify the following expression in $x$: \\[2x+8x^2+9-(4-2x-8x^2).\\] Express your answer in the form $ax^2 +bx+c$, where $a$, $b$, and $c$ are numbers.",
        "solution": """
To simplify the expression, we can use Python's sympy library.
{code_begin}from sympy import symbols, simplify

x = symbols('x')
expr = 2*x + 8*x**2 + 9 - (4 - 2*x - 8*x**2)

# Simplify the expression to get the canonical form
print(simplify(expr))
{code_end}{code_output_begin}
16*x**2 + 4*x + 5
{code_output_end}
So the simplified expression is $\\boxed{16x^2+4x+5}$
""".strip(),
    },
    {
        "problem": "What is the largest value of $x$ such that the expression \\[\\dfrac{x+1}{8x^2-65x+8}\\] is not defined?",
        "solution": """
To solve this problem, we can use Python's sympy library.
{code_begin}from sympy import Symbol, simplify, solve

x = Symbol('x')
expr = (x + 1)/(8*x**2 - 65*x + 8)

# To identify where the expression is invalid, we need to find the values of x for which the inverted expression is 0
# First invert the expression and simplify it
inverse_expr = simplify(1/expr)

# Solve for x for which inverse expression is 0
solutions = solve(inverse_expr, x)

# The solution asks for the largest value of $x$
max_soln = max(solutions)
max_soln
{code_end}{code_output_begin}
8
{code_output_end}
So the largest value of $x$ for which the expression is undefined is \\boxed{8}
""".strip(),
    },
    {
        "problem": "Let $t(x) = \\sqrt{3x+1}$ and $f(x)=5-t(x)$. What is $t(f(5))$?",
        "solution": """
To compute t(f(5)), we first need to compute f(5), which in turn requires computing t(5) first.
To compute this function composition, we use Python's sympy library.
{code_begin}from sympy import Symbol, sqrt

x = Symbol('x')
# Define the two functions
t_x = sqrt(3*x + 1)
f_x = 5 - t_x

# For computing t(f(5)), let's first compute f(5)
f_5 = f_x.subs(x, 5)

# Now use f_5 as input to expression for t_x
t_f_5 = t_x.subs(x, f_5)
t_f_5
{code_end}{code_output_begin}
2
{code_output_end}
So $t(f(5))$ is \\boxed{2}
""".strip(),
    },
    {
        "problem": "How many integers are solutions to the equation $$(x-2)^{(25-x^2)}=1?$$",
        "solution": """
Let's first solve the equation using sympy.
{code_begin}from sympy import symbols, Eq, solve

x = symbols('x')
# Define the equation
equation = Eq((x - 2)**(25 - x**2), 1)

# Solve the equation
roots = solve(equation, x)

# To count the number of integer solutions:
integer_solns = [isinstance(soln, int) for soln in roots]
len(integer_solns)
{code_end}{code_output_begin}
4
{code_output_end}
We have \\boxed{4} integer solutions.
""".strip(),
    },
]


math_text_with_code_probability = [
    {
        "problem": "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.",
        "solution": """
Since landing on the three parts are mutually exclusive events.
To compute the probability of landing on C, we can simply subtract the probability of landing on A and B.
{code_begin}from sympy import Rational

# probability of landing on A is 1/3
prob_A = Rational(1, 3)

# probability of landing on B is 5/12
prob_B = Rational(5, 12)

# Because A, B, and C are the only parts of the board, they cover the entire event space. Hence, their probabilities should sum to 1
prob_C = 1 - prob_A - prob_B

prob_C
{code_end}{code_output_begin}
1/4
{code_output_end}
So probability of the event is $\\boxed{\\frac{1}{4}}$.
""".strip(),
    },
    {
        "problem": "A box contains 5 white balls and 6 black balls.  Two balls are drawn out of the box at random.  What is the probability that they both are white?",
        "solution": """
To solve the problem, we can use Python's sympy library.
{code_begin}from sympy import Rational, binomial

# Number of white balls is 5
num_white_balls = 5

# Number of black balls is 6
num_black_balls = 6

# Total balls
total_balls = num_white_balls + num_black_balls

# Combinations of picking two white balls
comb_two_white_balls = binomial(num_white_balls, 2)

# Total combinations
total_comb = binomial(total_balls, 2)

# Probability of picking first two balls to be white
prob = comb_two_white_balls/total_comb

prob
{code_end}{code_output_begin}
2/11
{code_output_end}
So probability of the event is $\\boxed{\\frac{2}{11}}$.
""".strip(),
    },
    {
        "problem": "I have 5 marbles numbered 1 through 5 in a bag.  Suppose I take out two different marbles at random.  What is the expected value of the sum of the numbers on the marbles?",
        "solution": """
To calculate the expected value of the sum of two marbles, we need to iterate over all possible combinations.
To facilitate this computation, we will use Python's numerical libraries.
{code_begin}from itertools import combinations
from sympy import binomial

# List of numbers
numbers = [1, 2, 3, 4, 5]

total_pairs = binomial(5, 2)

# Initialize the sum of all pairs
all_sum_pair = 0

# Iterate over pairs of numbers
for pair in combinations(numbers, 2):
    # Increment total sum by sum of current pair
    all_sum_pair += sum(pair)

# Expected value is given by
all_sum_pair/total_pairs
{code_end}{code_output_begin}
6
{code_output_end}
So the expected value is \\boxed{6}
""".strip(),
    },
    {
        "problem": "How many different four-letter arrangements can be formed using the six letters $A, B, C, D, E$ and $F$, if the first letter must be $C$, one of the other letters must be $B$, and no letter can be used more than once in the arrangement?",
        "solution": """
Given that the first letter is C, to construct a four-letter word with unique characters, there are three remaining choices.
Out of these three letters, one of the letter has to be B.
To calculate the number of possible arrangements, we will use Python's sympy library.
{code_begin}from sympy import binomial

# Number of ways to make one of the remaining three letters B
choice_B = 3

# There are two other unique characters to be picked
# Total ways of choosing the first character which is different from C and B
choice_first = 6 - 2

# Total ways of choosing the last character different from the previous three
choice_last = 6 - 3

# Total arrangements
choice_B * choice_first * choice_last
{code_end}{code_output_begin}
36
{code_output_end}
So the total arrangements are \\boxed{36}
""".strip(),
    },
    {
        "problem": "Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction.",
        "solution": """
Melinda can never construct 10 and 20 because there's no 0 on a dice.
To construct numbers between 10 and 20, she needs a 1 on at least one of the dices.
To calculate these odds we will use Python's sympy library.
{code_begin}from sympy import Rational

# Probability that a dice gets something other than 1
prob_not_one = 1 - Rational(1, 6)

# Probability that both dice don't get 1
prob_both_not_one = prob_not_one * prob_not_one

# Probability that at least one dice gets 1
1 - prob_both_not_one
{code_end}{code_output_begin}
11/36
{code_output_end}
So the total arrangements are $\\boxed{\\frac{11}{36}}$.
""".strip(),
    },
]

math_text_with_code_intermediate_algebra = [
    {
        "problem": "What is the value of $\\left(1 - \\frac{1}{2}\\right)\\left(1-\\frac{1}{3}\\right)\\left(1-\\frac{1}{4}\\right)\\left(1-\\frac{1}{5}\\right) \\dotsm \\left(1-\\frac{1}{50}\\right)$? Express your answer as a common fraction.",
        "solution": """
To solve this problem, we can use Python's sympy library.
{code_begin}from sympy import Rational

product = 1
for i in range(2, 51):
    product *= (1 - Rational(1, i))

product
{code_end}{code_output_begin}
1/50
{code_output_end}
So the answer is $\\boxed{\\frac{1}{50}}$.
""".strip(),
    },
    {
        "problem": "Find the domain of the rational function $g(x) = \\frac{x^3-2x^2+4x+3}{x^2-4x+3}$. Express your answer as a union of intervals.",
        "solution": """
To find the domain of the rational function
$g(x) = \\frac{x^3 - 2x^2 + 4x + 3}{x^2 - 4x + 3}$,
we need to identify the values of $x$ for which the denominator is not equal to zero,
since division by zero is undefined.
Let's use sympy to find the values of $x$ that make the denominator zero and then determine the domain:
{code_begin}from sympy import symbols, solve, S

x = symbols('x')
denom = x**2 - 4*x + 3
zeros_of_denom = solve(denom, x)

# The domain of the function is the real numbers excluding the zeros of the denominator
domain = S.Reals - set(zeros_of_denom)
domain
{code_end}{code_output_begin}
Union(Interval.open(-oo, 1), Interval.open(1, 3), Interval.open(3, oo))
{code_output_end}
So the domain is: $\\boxed{ (-\\infty, 1) \\cup (1, 3) \\cup (3, \\infty) }$.
""".strip(),
    },
    {
        "problem": "If $2x+7$ is a factor of $6x^3+19x^2+cx+35$, find $c$.",
        "solution": """
We can solve this problem using sympy library.
{code_begin}from sympy import symbols, div, solve

# Define the symbols
x, c = symbols('x c')

# Define the polynomials
polynomial = 6*x**3 + 19*x**2 + c*x + 35
factor = 2*x + 7

# Perform polynomial division
quotient, remainder = div(polynomial, factor)

# Since the factor is a divisor, the remainder should be zero
# We can solve for c by setting the remainder to zero and solving for c
c_value = solve(remainder, c)
c_value
{code_end}{code_output_begin}
[3]
{code_output_end}
So the only possible value of c is $\\boxed{3}$.
""".strip(),
    },
    {
        "problem": "Find the minimum value of\n\\[\\frac{x^2}{x - 8}\\]for $x > 8.$",
        "solution": """
To solve this problem we can
- find critical points in the desired interval
- evaluate the function at critical points
- find the minimum among the values.
To do the calculations we'll use sympy.
{code_begin}from sympy import symbols, diff, solveset, Interval

# Define the symbol
x = symbols('x')

# Define the function
f = x**2 / (x - 8)

# First derivative of the function
f_prime = diff(f, x)

# Find the critical points in the interval x > 8
critical_points = solveset(f_prime, x, domain=Interval.open(8, float('inf')))

# Evaluate the function at the critical points to find the minimum value
values = [f.subs(x, cp).evalf() for cp in critical_points if cp.is_real]
values
{code_end}{code_output_begin}
[32.0000000000000]
{code_output_end}
Thus the minimum value of the given function for $x > 8$ is $\\boxed{32}$.
""".strip(),
    },
    {
        "problem": "Two of the roots of\n\\[ax^3 + (a + 2b) x^2 + (b - 3a) x + (8 - a) = 0\\]are $-2$ and 3.  Find the third root.",
        "solution": """
To find the third root of the given cubic equation
$ax^3 + (a + 2b)x^2 + (b - 3a)x + (8 - a) = 0$,
we first need to determine the values of the coefficients $a$ and $b$.
Given that two of the roots are -2 and 3, we can substitute these values into the equation:
{code_begin}from sympy import symbols, Eq, solve

# Define the symbols
a, b, x = symbols('a b x')

eq = a * x**3 + (a + 2 * b) * x**2 + (b - 3 * a) * x + (8 - a)

# Form two equations substituting x by equation roots
eq1 = eq.subs(x, -2)
eq2 = eq.subs(x, 3)

# Solving the equations
solution = solve((eq1, eq2), (a, b))
solution
{code_end}{code_output_begin}
{a: 8/9, b: -40/27}
{code_output_end}
By Vieta's formulas the sum of the roots of a cubic equation
$ax^3 + bx^2 + cx + d = 0$ is equal to $-\frac{b}{a}$.
In our case, the equation is
$ax^3 + (a + 2b)x^2 + (b - 3a)x + (8 - a) = 0$.
Hence, the sum of the roots is $-\frac{a + 2b}{a}$.
Since two of the roots are -2 and 3, let the third root be $r$. Thus, we have:
$-2 + 3 + r = -\frac{a + 2b}{a}$.
Let's solve this in sympy:
{code_begin}r = symbols('r')
sum_of_roots_eq = Eq(-2 + 3 + r, -(a + 2*b)/a)

# Substitute a and b values
sum_of_roots_eq_substituted = sum_of_roots_eq.subs(solution)

# Solve for r
third_root = solve(sum_of_roots_eq_substituted, r)
third_root
{code_end}{code_output_begin}
[4/3]
{code_output_end}
The third root of the given cubic equation is $\\boxed{\\frac{4}{3}}$.
""".strip(),
    },
]

math_text_with_code_number_theory = [
    {
        "problem": "The integer $m$ is between $30$ and $80$ and is a multiple of $6$. When $m$ is divided by $8,$ the remainder is $2$. Similarly, when $m$ is divided by $5,$ the remainder is $2$. What is the value of $m$?",
        "solution": """
Since the interval is relatively small, we can check all the values from 30 to 80:
{code_begin}# Iterate through possible values of m between 30 and 80 (inclusive)
for m in range(30, 81):
    if m % 6 == 0 and m % 8 == 2 and m % 5 == 2:
        print(m)
{code_end}{code_output_begin}
42
{code_output_end}
The only number that satisfies the requirements is $\\boxed{42}$.
""".strip(),
    },
    {
        "problem": "In a physical education class, students line up in four rows to do exercises. One particular class has over 30 students, with three rows having the same number of students and one row having one more student than the other three rows. What is the smallest possible class size for this physical education class?",
        "solution": """
If $x$ is a number of students in one row, then total number of students is $3x + (x + 1)$.
This should be greater or equals to 30: $3x + (x + 1) \\ge 30$
{code_begin}from sympy import symbols, solve

x = symbols('x')
# Define the inequality
inequality = 3*x + (x + 1) >= 30

# Solve the inequality
solution = solve(inequality, x)
solution
{code_end}{code_output_begin}
(29/4 <= x) & (x < oo)
{code_output_end}
So the minimum number of students in a row is $29/4 = 7.25$.
Since this must be an integer, such minimum number is $8$.
Thus the total number of students is
{code_begin}row_students = 8
total_students = 3 * row_students + (row_students + 1)
total_students
{code_end}{code_output_begin}
33
{code_output_end}
The answer is $\\boxed{33}$.
""".strip(),
    },
    {
        "problem": "How many more digits does the base-3 representation of $987_{10}$ have than the base-8 representation of $987_{10}$?",
        "solution": """
We can easily do it in python.
First, we need to create a function that changes base-10 numbers to other bases.
Then we get base-3 and base-8 representations and calculate lengths difference.
{code_begin}def change_base(num, base=10):
    if num == 0:
        return '0'
    nums = []
    while num:
        num, r = divmod(num, base)
        nums.append(str(r))
    return ''.join(reversed(nums))

# Get the base-3 and base-8 representation of 987
base_3_repr = change_base(num=987, base=3)
base_8_repr = change_base(num=987, base=8)

# difference in representation length
repr_diff = len(base_3_repr) - len(base_8_repr)
repr_diff
{code_end}{code_output_begin}
3
{code_output_end}
So the answer is $\\boxed{3}$.
""".strip(),
    },
    {
        "problem": "A two-digit integer $AB$ equals $\\frac{1}{9}$ of the three-digit integer $AAB$, where $A$ and $B$ represent distinct digits from 1 to 9. What is the smallest possible value of the three-digit integer $AAB$?",
        "solution": """
Let's iterate over all possible values of $a$ and $b$ and check what is the smallest value that satisfies conditions.
{code_begin}for a in range(1, 10):
    for b in range(1, 10):
        if a == b:
            continue
        first_num_val = 10 * a + b
        second_num_val = 100 * a + 10 * a + b

        if first_num_val * 9 == second_num_val:
            # We found the smallest AAB which satisfies all the constraints
            print(second_num_val)
            break
{code_end}{code_output_begin}
225
{code_output_end}
Thus the smallest such number is $\\boxed{225}$
""",
    },
    {
        "problem": "How many of the positive divisors of 3240 are multiples of 3?",
        "solution": """
We can list all divisors of 3240 and filter only multiples of 3.
Let's use sympy for this:
{code_begin}from sympy import divisors

# Calculate all the divisors of 3240
all_divisors = divisors(3240)

# Filter the divisors that are multiples of 3
divisors_multiple_of_3 = [d for d in all_divisors if d % 3 == 0]

# Count the number of divisors that are multiples of 3
number_of_divisors_multiple_of_3 = len(divisors_multiple_of_3)
number_of_divisors_multiple_of_3
{code_end}{code_output_begin}
32
{code_output_end}
So 3240 has $\\boxed{32}$ divisors that are multiples of 3.
""".strip(),
    },
]

math_text_with_code_geometry = [
    {
        "problem": "In an isosceles triangle, one of the angles opposite an equal side is $40^{\\circ}$. How many degrees are in the measure of the triangle's largest angle?",
        "solution": """
In an isosceles triangle two angles opposite the equal sides are equal.
Let A and B be these angles.
We know that the sum of all angles is 180 degrees.
Thus A + B + C = 180.
Let's solve this using sympy:
{code_begin}from sympy import symbols, Eq, solve

A = 40
B = A
C = symbols('C')

eq = Eq(A + B + C, 180)
C_value = solve(eq)[0]

A, B, C_value
{code_end}{code_output_begin}
(40, 40, 100)
{code_output_end}
Since we need to find the measure of the largest angle, the answer is $\\boxed{100}$.
""".strip(),
    },
    {
        "problem": "Point $A$ has coordinates $(x,6)$. When Point $A$ is reflected over the $y$-axis it lands on Point $B$. What is the sum of the four coordinate values of points $A$ and $B$?",
        "solution": """
Reflection over the $y$-axis changes $x$ coordinate to $-x$ and keeps $y$ coordinate constant.
We can calculate the sum of the four coordinate values using sympy library:
{code_begin}from sympy import symbols

# Define the symbol
x = symbols('x')

A = (x, 6)
B = (-x, 6)

sum_coordinates = sum(A) + sum(B)
sum_coordinates
{code_end}{code_output_begin}
12
{code_output_end}
Thus the answer is $\\boxed{12}$.
""".strip(),
    },
    {
        "problem": "Kadin makes a snowman by stacking snowballs of radius 2 inches, 3 inches, and 5 inches.  Assuming all his snowballs are spherical, what is the total volume of snow he uses, in cubic inches?  Express your answer in terms of $\\pi$.",
        "solution": """
To calculate the total volume of a snowman we need to sum volumes of three snowballs.
The volume of a sphere with radius $r$ is $\\frac{4}{3}\\pi r^3$.
We can use sympy for the calculations:
{code_begin}from sympy import pi, Rational

# Radii of the snowballs
r1 = 2
r2 = 3
r3 = 5

# Calculate the volume of each snowball
V1 = Rational(4, 3) * pi * r1**3
V2 = Rational(4, 3) * pi * r2**3
V3 = Rational(4, 3) * pi * r3**3

# Total volume of the snowman
total_volume = V1 + V2 + V3
total_volume
{code_end}{code_output_begin}
640*pi/3
{code_output_end}
So the total volume is $\\boxed{\\frac{640}{3}\\pi}$.
""".strip(),
    },
    {
        "problem": "A fly is on the edge of a ceiling of a circular room with a radius of 58 feet. The fly walks straight across the ceiling to the opposite edge, passing through the center of the circle. It then walks straight to another point on the edge of the circle but not back through the center. The third part of the journey is straight back to the original starting point. If the third part of the journey was 80 feet long, how many total feet did the fly travel over the course of all three parts?",
        "solution": """
Let x be the unknown second part of the journey and d be the diameter of the circle.
The journey forms a triangle with sides d, x, and 80.
Since this triangle is inscribed in a circle and one of its sides is a diameter,
this triangle is right-angled. So we can apply Pythagorean theorem:
$x^2 + 80^2 = d^2$.
We can solve this equation using sympy:
{code_begin}from sympy import symbols, Eq, solve

# Radius and diameter of the circular room
r = 58
d = 2 * r

x = symbols('x')
eq = Eq(x ** 2 + 80 ** 2, d ** 2)

# We need only positive solutions so take maximum
x_value = max(solve(eq))

# Calculate the total distance traveled as the sum of all three parts
total_feet = x_value + d + 80
total_feet
{code_end}{code_output_begin}
280
{code_output_end}
So the total distance traveled by the fly is $\\boxed{280}$.
""".strip(),
    },
    {
        "problem": "A rectangular box $P$ is inscribed in a sphere of radius $r$. The surface area of $P$ is 384, and the sum of the lengths of its 12 edges is 112. What is $r$?",
        "solution": """
To solve this problem, we need to use the properties of a rectangular box and a sphere.
The box is inscribed in the sphere, which means that the sphere's diameter is the diagonal of the box.
Assuming $x$, $y$, and $z$ to be dimensions of a box, we can make three equations.
One for the sum of edges, one for the surface area and one for the sphere diameter.
- $2xy + 2yz + 2xz = 384$
- $4 * (x + y + z) = 112$
- $x ^ 2 + y ^ 2 + z ^ 2 = r ^ 2$

Let's use sympy to solve them.
{code_begin}from sympy import symbols, Eq, solve

# Define the variables
x, y, z, r = symbols('x y z r')

# Equation for the surface area
eq1 = Eq(2 * (x*y + x*z + y*z), 384)
# Equation for the length of 12 edges
eq2 = Eq(4 * (x + y + z), 112)
# Equation for the diameter of the sphere
eq3 = Eq(x**2 + y**2 + z**2, (2*r)**2)


# Solve the equations
solutions = solve((eq1, eq2, eq3), (x, y, z, r))
solutions
{code_end}{code_output_begin}
[(-z/2 - sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  -z/2 + sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  z,
  -10),
 (-z/2 + sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  -z/2 - sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  z,
  -10),
 (-z/2 - sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  -z/2 + sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  z,
  10),
 (-z/2 + sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  -z/2 - sqrt(-3*z**2 + 56*z + 16)/2 + 14,
  z,
  10)]
{code_output_end}
We need the solution corresponding to the r: the last value of tuple.
And we need r to be positive.
So the radius of the sphere $r$ is $\\boxed{10}$.
""".strip(),
    },
]

math_text_with_code_precalculus = [
    {
        "problem": "If $\\det \\mathbf{M} = -2,$ then find $ \\det (\\mathbf{M}^4).$",
        "solution": """
We know that det(AB) = det(A) * det(B).
We will use this fact to compute $\\det (\\mathbf{M}^4)$ using Python code.
{code_begin}det_M = -2

# Determinant of M^4
det_M_4 = det_M**4

det_M_4
{code_end}{code_output_begin}
16
{code_output_end}
So the answer is \\boxed{16}.
""".strip(),
    },
    {
        "problem": "If $\\sin x = 3 \\cos x,$ then what is $\\sin x \\cos x$?",
        "solution": """
We know that $\\sin^2 x + \\cos^2 x = 1.$
Substituting $\\sin x = 3 \\cos x$ in the above equation, we get  $9 \\cos^2 x + \\cos^2 x = 1$
Thus, $\\cos^2 x = \\frac{1}{10}$.
Since $\\sin x \\cos x = 3 \\cos x \\cos x = 3 \\cos^2 x$.
Substituting $\\cos^2 x = \\frac{1}{10}$, we get the $\\sin x \\cos x = \\frac{3}{10}$.
Hence, the answer is $\\boxed{\\frac{3}{10}}$.
""".strip(),
    },
    {
        "problem": "Let $\\mathbf{a} = \\begin{pmatrix} 2 \\\\ 1 \\\\ 5 \\end{pmatrix}.$  Find the vector $\\mathbf{b}$ such that $\\mathbf{a} \\cdot \\mathbf{b} = 11$ and\n\\[\\mathbf{a} \\times \\mathbf{b} = \\begin{pmatrix} -13 \\\\ -9 \\\\ 7 \\end{pmatrix}.\\]",
        "solution": """
Using the dot product and cross product constraints, we can solve the problem using the sympy library.
{code_begin}from sympy import symbols, Eq, solve
from sympy.vector import CoordSys3D

# Define the symbolic variables and the coordinate system
b1, b2, b3 = symbols('b1 b2 b3')
N = CoordSys3D('N')

# Define the vectors
a = 2*N.i + 1*N.j + 5*N.k
b = b1*N.i + b2*N.j + b3*N.k

# Define the dot product equation
dot_product_eq = Eq(a.dot(b), 11)

# Define the cross product equations
cross_product = a.cross(b)
# Cross product is [-13, -9, 7]
first_dim_eq = Eq(cross_product.dot(N.i), -13)
second_dim_eq = Eq(cross_product.dot(N.j), -9)
third_dim_eq = Eq(cross_product.dot(N.k), 7)

# Solve the system of equations
solution = solve([dot_product_eq, first_dim_eq, second_dim_eq, third_dim_eq], [b1, b2, b3])
print([solution[b1], solution[b2], solution[b3]])
{code_end}{code_output_begin}
[-1, 3, 2]
{code_output_end}
So the answer is $\\boxed{\\begin{pmatrix} -1 \\\\ 3 \\\\ 2 \\end{pmatrix}}$.
""".strip(),
    },
    {
        "problem": "Simplify $\\tan \\frac{\\pi}{24} + \\tan \\frac{7 \\pi}{24}.$",
        "solution": """
We will use sympy's simplify function to simplify the given trigonometric expression.
{code_begin}from sympy import tan, pi, simplify

# Define the angles in radians
angle1 = pi/24
angle2 = 7*pi/24

# Define the expression
expression = tan(angle1) + tan(angle2)

# Simplify the expression
simplified_expression = simplify(expression)

print(simplified_expression)
{code_end}{code_output_begin}
-2*sqrt(3) + 2*sqrt(6)
{code_output_end}
So the answer is $\\boxed{2 \\sqrt{6} - 2 \\sqrt{3}}$.
""".strip(),
    },
    {
        "problem": "A certain regular tetrahedron has three of its vertices at the points $(0,1,2),$ $(4,2,1),$ and $(3,1,5).$  Find the coordinates of the fourth vertex, given that they are also all integers.",
        "solution": """
The key insight to finding the missing vertex is that a regular tetrahedron has all the sides of equal length.
Using the three provided vertices, we can create three equations constraining the fourth vertex and solve them to get solutions.
Here' the solution in Python's sympy library.
{code_begin}from sympy import symbols, Eq, solve
from sympy.vector import CoordSys3D

# Define the coordinate system
N = CoordSys3D('N')

# Define the vertices of the tetrahedron
A = 0*N.i + 1*N.j + 2*N.k
B = 4*N.i + 2*N.j + 1*N.k
C = 3*N.i + 1*N.j + 5*N.k

# Variables for the fourth vertex D
x, y, z = symbols('x y z')
D = x*N.i + y*N.j + z*N.k

# For a regular tetrahedron, the distance between any two vertices is the same
dist_AB = (B - A).magnitude()

# Equations based on the distances AD, BD, and CD being equal to AB
eq_AD = Eq((D - A).magnitude(), dist_AB)
eq_BD = Eq((D - B).magnitude(), dist_AB)
eq_CD = Eq((D - C).magnitude(), dist_AB)

# Solving the system of equations for x, y, z
solutions = solve((eq_AD, eq_BD, eq_CD), (x, y, z))

# Filtering solutions for integer coordinates
integer_solutions = [sol for sol in solutions if all(coord.is_Integer for coord in sol)]
integer_solutions
{code_end}{code_output_begin}
[(3, -2, 2)]
{code_output_end}
Hence the fourth vertex is \\boxed{(3,-2,2)}.
""".strip(),
    },
]

math_text_with_code_prealgebra = [
    {
        "problem": 'Six students participate in an apple eating contest. The graph shows the number of apples eaten by each participating student. Aaron ate the most apples and Zeb ate the fewest. How many more apples than Zeb did Aaron eat?\n\n[asy]\ndefaultpen(linewidth(1pt)+fontsize(10pt));\npair[] yaxis = new pair[8];\nfor( int i = 0 ; i < 8 ; ++i ){\n\nyaxis[i] = (0,i);\n\ndraw(yaxis[i]--yaxis[i]+(13,0));\n}\ndraw((0,0)--(0,7));\ndraw((13,7)--(13,0));\n\nfill((1,0)--(1,2)--(2,2)--(2,0)--cycle,grey);\nfill((3,0)--(3,3)--(4,3)--(4,0)--cycle,grey);\nfill((5,0)--(5,6)--(6,6)--(6,0)--cycle,grey);\nfill((7,0)--(7,4)--(8,4)--(8,0)--cycle,grey);\nfill((9,0)--(9,3)--(10,3)--(10,0)--cycle,grey);\nfill((11,0)--(11,1)--(12,1)--(12,0)--cycle,grey);\nlabel("0",yaxis[0],W);\nlabel("1",yaxis[1],W);\nlabel("2",yaxis[2],W);\nlabel("3",yaxis[3],W);\nlabel("4",yaxis[4],W);\nlabel("5",yaxis[5],W);\nlabel("6",yaxis[6],W);\nlabel("7",yaxis[7],W);\nlabel("Students/Participants",(6.5,0),S);\nlabel("Results of an Apple Eating Contest",(6.5,7),N);\nlabel(rotate(90)*"$\\#$ of Apples Eaten",(-1,3.5),W);\n[/asy]',
        "solution": """
There are a total of six columns in the graph with values: 2, 3, 6, 4, 3, and 1.
Since Aaron ate the most apples, and Zeb ate the fewest apples, we can answer the question using the following Python code.
{code_begin}apple_count = [2, 3, 6, 4, 3, 1]

# Aaron apple count
aaron_apple = max(apple_count)

# Zeb apple count
zeb_apple = min(apple_count)

diff_apple = aaron_apple - zeb_apple
diff_apple
{code_end}{code_output_begin}
5
{code_output_end}
So Aaron ate \\boxed{5} more apples than Zeb.
""".strip(),
    },
    {
        "problem": "I have a bag with only red, blue, and green marbles.  The ratio of red marbles to blue marbles to green marbles is $1:5:3$.  There are 27 green marbles in the bag.  How many marbles are there  in the bag?",
        "solution": """
To compute the total number of marbles, we can use Python's sympy library.
{code_begin}from sympy import Symbol, Eq, solve

# Let the number of red marbles be represented by 'red'
red = Symbol('red')
# Number of red to blue to green is 1:5:3
blue = 5 * red
green = 3 * red

# Total marbles are just the sum of the three colors
total = red + blue + green

# Given the count of green marbles, we can solve for red marbles
red_count = solve(Eq(green, 27), red)[0]
# And substitute the
total_count = total.subs(red, red_count)
total_count
{code_end}{code_output_begin}
81
{code_output_end}
Hence, there are a total of \\boxed{81} marbles.
""".strip(),
    },
    {
        "problem": "How many perfect cubes are between 100 and 900?",
        "solution": """
To calculate these perfect cubes between 100 and 900, there are a few ways.
One way is to enumerate cubes, and calculate the ones which satisfy this constraint.
Another is to just take cube roots of the interval endpoints, with ceil and floor fns applied to start and end of the boundary.
To solve the problem using the latter strategy, we use the following Python code.
{code_begin}import math

lower_val = int(math.ceil(100**(1/3)))
higher_val = int(math.floor(900**(1/3)))

num_perfect_cubes = higher_val - lower_val + 1
print(num_perfect_cubes)
{code_end}{code_output_begin}
5
{code_output_end}
Hence, there are a total of \\boxed{5} perfect cubes between 100 and 900.
""".strip(),
    },
    {
        "problem": "Find the smallest positive integer $x$ which is greater than $1$ and relatively prime to $120$ (recall that relatively prime means that the GCD of $x$ and $120$ is $1$)",
        "solution": """
To compute the smallest positive integer which is relatively prime to 120, we can iterate over all the numbers greater than 1 and less than 120, and break whenever we find a number which has a gcd of 1 with 120.
{code_begin}from sympy import gcd

answer = None
# Limiting our search space to 119
for number in range(2, 120):
    if gcd(number, 120) == 1:
        answer = number
        break

print(answer)
{code_end}{code_output_begin}
7
{code_output_end}
Hence, \\boxed{7} is the smallest number which is relatively prime to 120.
""".strip(),
    },
    {
        "problem": "In square $ABCD$, point $M$ is the midpoint of side $AB$ and point $N$ is the midpoint of side $BC$. What is the ratio of the area of triangle $AMN$ to the area of square $ABCD$? Express your answer as a common fraction.",
        "solution": """
Let each side of the square be 1. So the area of ABCD is 1^2=1.
The triangle AMN, has base of length AM=1/2, and height of length of 1/2 as well since BN=1/2.
Hence, area of AMN = (1/2) * base * height = (1/2) * (1/2) * (1/2) = 1/8.
Hence. ratio of area of AMN to ABCD is $\\boxed{\\frac{1}{8}}$.
""".strip(),
    },
]


math_problem_augmentation = [
    {
        "problem": """
In the equation
$$5x^2-kx+1=0$$
determine $k$ such that the difference of the roots be equal to unity.
""".strip(),
        "augmented_problem": """
The roots $x_1$ and $x_2$ of the equation
$$x^2-3ax+a^2=0$$
are such that
$x_1^2+x_2^2=1.75$.
Determine $a$.
""".strip(),
    },
    {
        "problem": """
Solve the following equation
$\\ds\\f{3+x}{3x}=\\sqrt {\\ds\\f{1}{9}+\\ds\\f{1}{x}\\sqrt {\\ds\\f{4}{9}+\\ds\\f{2}{x^2}}}$
""".strip(),
        "augmented_problem": """
Solve the following equation
$\\sqrt {1+x\\sqrt {x^2+24}}=x+1$
""".strip(),
    },
    {
        "problem": """
In an infinitely decreasing geometric progression the sum
of all the terms occupying odd places is equal to 36, and that of all
the terms at even places equals 12.
Find the progression.
""".strip(),
        "augmented_problem": """
The sum of the terms of an infinitely decreasing geometric
progression is equal to 56, and the sum of the squared terms of the
same progression is 448.
Find the first term and the common ratio.
""".strip(),
    },
    {
        "problem": """
Two railway stations are at a distance of 96 km from each other.
One train covers this distance 40 minutes faster than does the other.
The speed of the first train is 12 km/h higher than that of the second.
Determine the speed of both trains.
""".strip(),
        "augmented_problem": """
A student was asked to multiply 78 by a two-digit number
in which the tens digit was three times as large as the units digit;
by mistake, he interchanged the digits in the second factor and
thus obtained a product smaller than the true product by 2808.
What was the true product?
""".strip(),
    },
]


examples_map = {
    "math_standard_few_shot": math_standard_few_shot,
    "math_text_with_code": math_text_with_code,
    "math_text_detailed": math_text_detailed,
    "math_problem_augmentation": math_problem_augmentation,
    # 7 subjects
    "math_text_with_code_algebra": math_text_with_code_algebra,
    "math_text_with_code_probability": math_text_with_code_probability,
    "math_text_with_code_intermediate_algebra": math_text_with_code_intermediate_algebra,
    "math_text_with_code_number_theory": math_text_with_code_number_theory,
    "math_text_with_code_geometry": math_text_with_code_geometry,
    "math_text_with_code_precalculus": math_text_with_code_precalculus,
    "math_text_with_code_prealgebra": math_text_with_code_prealgebra,
}
