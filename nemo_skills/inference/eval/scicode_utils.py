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

# adapted from https://github.com/scicode-bench/SciCode/blob/main/eval/scripts/gencode.py

import logging
import re

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def process_problem_code(prob_data: dict, num_steps: int) -> str:
    header_docstring = prob_data["sub_steps"][num_steps]["function_header"]
    return_str = prob_data["sub_steps"][num_steps]["return_line"]
    string = f"{header_docstring}\n\n{return_str}"
    return string


def process_problem_steps(problem_data: dict, num_steps: int, previous_llm_code, with_background):
    """Process problem data and return previous steps and next steps"""
    output_lines = []
    next_step = []
    previous_code = []
    for i in range(num_steps):
        output_lines.append(
            problem_data["sub_steps"][i]["step_description_prompt"]
            + "\n"
            + problem_data["sub_steps"][i]["step_background"]
            if with_background
            else problem_data["sub_steps"][i]["step_description_prompt"]
        )
        output_lines.append(previous_llm_code[i])
        previous_code.append(previous_llm_code[i])
        output_lines.append("------")

    next_step.append(
        problem_data["sub_steps"][num_steps]["step_description_prompt"]
        + "\n"
        + problem_data["sub_steps"][num_steps]["step_background"]
        if with_background
        else problem_data["sub_steps"][num_steps]["step_description_prompt"]
    )
    next_step.append(process_problem_code(problem_data, num_steps))
    output_str = "\n\n".join(output_lines[:-1])  # Remove the last "------"
    next_step_str = "\n\n".join(next_step)
    previous_code_str = "\n".join(previous_code)
    return output_str, next_step_str, previous_code_str


def extract_python_script(response: str):
    # We will extract the python script from the response
    if "```" in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if "```python" in response
            else response.split("```")[1].split("```")[0]
        )
    else:
        LOG.info("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE)
    return python_script


# there is a weird if condition in scicode codebase that prefills
# this code for those problems instead of generating with an llm
# not fully sure what's the reason, but following the original implementation

prefilled_steps_code = {
    (
        "13",
        5,
    ): '''
def __init__(self, n_grid, x_out):
    """Constructor sets up coordinates, memory for variables.
        The variables:
            mesh points:
                x: the x coordinate for each mesh grid
                y: the y coordinate for each mesh grid
                z: the z coordinate for each mesh grid
                t: the time coordinate of the simulation
                r: the distance to the origin for each mesh grid
            evolving fields:
                E_x: the x component of the field E
                E_y: the y componnet of the field E
                E_z: the z component of the field E
                A_x: the x component of the field A
                A_y: the y component of the field A
                A_z: the z component of the field A
                phi: the scalar potential field phi values
            monitor variables:
                constraint: the current constraint violation value from the evolving fields.

        """
    self.n_grid = n_grid
    self.n_vars = 7
    self.delta = float(x_out) / (n_grid - 2.0)
    delta = self.delta
    self.x = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[:, None, None]
    self.y = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[None, :, None]
    self.z = np.linspace(-self.delta * 0.5, x_out + 0.5 * self.delta, self.n_grid)[None, None, :]
    self.r = np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    self.E_x = zeros((n_grid, n_grid, n_grid))
    self.E_y = zeros((n_grid, n_grid, n_grid))
    self.E_z = zeros((n_grid, n_grid, n_grid))
    self.A_x = zeros((n_grid, n_grid, n_grid))
    self.A_y = zeros((n_grid, n_grid, n_grid))
    self.A_z = zeros((n_grid, n_grid, n_grid))
    self.phi = zeros((n_grid, n_grid, n_grid))
    self.constraint = zeros((n_grid, n_grid, n_grid))
    self.t = 0.0
'''.strip(),
    (
        "62",
        0,
    ): """
def __init__(self, length, basis_size, operator_dict):
    self.length = length
    self.basis_size = basis_size
    self.operator_dict = operator_dict
""".strip(),
    (
        "76",
        2,
    ): '''
def generate_dna(N: int, PWM: dict) -> tuple:
    """
    Input:
    N (int): Length of the resultant DNA sequence.
    PWM matrix with keys 'A', 'C', 'G', 'T'

    Output:
    tuple: Insertion location (int), DNA sequence (str), DNA reverse complement (str)
    """
    p = random.randint(0, N - 1)
    nucleotide = 'ACGT'
    uni_weights = [0.25, 0.25, 0.25, 0.25]
    dna_string = ''.join(random.choices(nucleotide, uni_weights, k=N))
    spike_mat = load_motif_from_df(PWM)
    spiked_seq = ''.join((random.choices(nucleotide, weights=[PWM[nuc][i] for nuc in nucleotide], k=1)[0] for i in range(len(PWM['A']))))
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    reversed_seq = dna_string[::-1]
    reverse_complement = ''.join((complement[nuc] for nuc in reversed_seq if nuc in complement))
    new_seq = dna_string[:p] + spiked_seq + dna_string[p:]
    new_seq_rc = reverse_complement[:N - p] + spiked_seq + reverse_complement[N - p:]
    return (p, new_seq, new_seq_rc)
'''.strip(),
}


eval_prefix = """
import h5py
import scipy
H5PY_FILE = "/data/test_data.h5"

def process_hdf5_list(group):
    lst = []
    for key in group.keys():
        lst.append(group[key][()])
    return lst


def process_hdf5_dict(group):
    dict = {}
    for key, obj in group.items():
        if isinstance(obj, h5py.Group):
            dict[key] = process_hdf5_sparse_matrix(obj['sparse_matrix'])
        elif isinstance(obj[()], bytes):
            dict[key] = obj[()].decode('utf-8', errors='strict')
        else:
            try:
                tmp = float(key)
                dict[tmp] = obj[()]
            except ValueError:
                dict[key] = obj[()]
    return dict


def process_hdf5_sparse_matrix(group):
    data = group['data'][()]
    shape = tuple(group['shape'][()])
    if 'row' in group and 'col' in group:
        row = group['row'][()]
        col = group['col'][()]
        return scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
    elif 'blocksize' in group:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        blocksize = tuple(group['blocksize'][()])
        return scipy.sparse.bsr_matrix((data, indices, indptr), shape=shape, blocksize=blocksize)
    else:
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def process_hdf5_datagroup(group):
    for key in group.keys():
        if key == "list":
            return process_hdf5_list(group[key])
        if key == "sparse_matrix":
            return process_hdf5_sparse_matrix(group[key])
        else:
            return process_hdf5_dict(group)


def process_hdf5_to_tuple(step_id, test_num, h5py_file=H5PY_FILE):
    data_lst = []
    with h5py.File(h5py_file, 'r') as f:
        for test_id in range(test_num):
            group_path = f'{step_id}/test{test_id + 1}'
            if isinstance(f[group_path], h5py.Group):
                group = f[group_path]  # test1, test2, test3
                num_keys = [key for key in group.keys()]
                if len(num_keys) == 1:  # only 1 var in the test
                    subgroup = group[num_keys[0]]
                    if isinstance(subgroup, h5py.Dataset):
                        if isinstance(subgroup[()], bytes):
                            data_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                        else:
                            data_lst.append(subgroup[()])
                    elif isinstance(subgroup, h5py.Group):
                        data_lst.append(process_hdf5_datagroup(subgroup))
                else:
                    var_lst = []
                    for key in group.keys():  # var1, var2, var3
                        subgroup = group[key]
                        if isinstance(subgroup, h5py.Dataset):
                            if isinstance(subgroup[()], bytes):
                                var_lst.append(subgroup[()].decode('utf-8', errors='strict'))
                            else:
                                var_lst.append(subgroup[()])
                        elif isinstance(subgroup, h5py.Group):
                            var_lst.append(process_hdf5_datagroup(subgroup))
                    data_lst.append(tuple(var_lst))
            else:
                raise FileNotFoundError(f'Path {group_path} not found in the file.')
    return data_lst


# Local comparison helpers adapted from SciCode src/scicode/compare/cmp.py
# Adding import here because function might use these libraries which might not be part of required_dependencies, for example: 17.1 -   "required_dependencies": "import sympy as sp; import numpy as np" - causes error:
# "Traceback (most recent call last): assert cmp_tuple_or_list(init_eji_array(10,[4,6,8,10]), target)    if not are_dicts_close(v1, v2):    dict1 = process_symbol_in_dict(dict1)    if isinstance(value, sympy.Symbol): NameError: name 'sympy' is not defined",
def are_dicts_close(dict1, dict2, atol=1e-8, rtol=1e-5):
    import sympy
    import scipy
    import numpy as np
    dict1 = process_symbol_in_dict(dict1)
    dict2 = process_symbol_in_dict(dict2)
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        value1 = dict1[key]
        value2 = dict2[key]
        if isinstance(value1, (sympy.Symbol, str)):
            if not value1 == value2:
                return False
        elif isinstance(value1, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix, scipy.sparse.bsr_matrix, scipy.sparse.coo_matrix)):
            value1 = value1.toarray()
            value2 = value2.toarray()
            if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                return False
        else:
            try:
                if not np.allclose(value1, value2, atol=atol, rtol=rtol):
                    return False
            except ValueError:
                if not value1 == value2:
                    return False
    return True


def process_symbol_in_dict(dict):
    import sympy
    new_dict = {}
    for key, value in dict.items():
        new_dict[key] = value
        if isinstance(value, sympy.Symbol):
            new_dict[key] = str(value)
        if isinstance(key, sympy.Symbol):
            new_dict[str(key)] = dict[key]
            new_dict.pop(key)
    return new_dict


def are_csc_matrix_close(matrix1, matrix2):
    import numpy as np
    dense1 = matrix1.toarray()
    dense2 = matrix2.toarray()
    return np.allclose(dense1, dense2)


def cmp_tuple_or_list(var1, var2):
    import scipy
    import numpy as np
    if len(var1) != len(var2):
        return False
    for v1, v2 in zip(var1, var2):
        if isinstance(v1, dict):
            if not are_dicts_close(v1, v2):
                return False
        elif isinstance(v1, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix)):
            if not are_csc_matrix_close(v1, v2):
                return False
        elif isinstance(v1, bool):
            if not v1 == v2:
                return False
        else:
            try:
                if not np.allclose(v1, v2):
                    return False
            except ValueError as e:
                print(e)
                if not v1 == v2:
                    return False
    return True
"""
