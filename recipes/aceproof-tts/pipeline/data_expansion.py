def expand_for_proof_generation(problems, n_parallel_proof_gen, interleave=False):
    expanded = []
    if interleave:
        for seed_idx in range(n_parallel_proof_gen):
            for problem in problems:
                problem_idx = problem["problem_idx"]
                row = dict(problem)
                row["generation_seed"] = seed_idx
                row["row_id"] = f"{problem_idx}_{seed_idx}"
                expanded.append(row)
    else:
        for problem in problems:
            problem_idx = problem["problem_idx"]
            for seed_idx in range(n_parallel_proof_gen):
                row = dict(problem)
                row["generation_seed"] = seed_idx
                row["row_id"] = f"{problem_idx}_{seed_idx}"
                expanded.append(row)
    return expanded


def expand_for_verification(proof_rows, n_verification_per_proof, interleave=False):
    expanded = []
    if interleave:
        for seed_idx in range(n_verification_per_proof):
            for proof_row in proof_rows:
                row_id = proof_row.get("row_id", proof_row.get("proof_id", "proof"))
                row = dict(proof_row)
                row["verification_seed"] = seed_idx
                row["verify_row_id"] = f"{row_id}_v{seed_idx}"
                expanded.append(row)
    else:
        for proof_row in proof_rows:
            row_id = proof_row.get("row_id", proof_row.get("proof_id", "proof"))
            for seed_idx in range(n_verification_per_proof):
                row = dict(proof_row)
                row["verification_seed"] = seed_idx
                row["verify_row_id"] = f"{row_id}_v{seed_idx}"
                expanded.append(row)
    return expanded


def expand_for_refinement(tasks, n_samples_per_trial, interleave=False):
    expanded = []
    if interleave:
        for sample_idx in range(n_samples_per_trial):
            for task in tasks:
                problem_idx = task["problem_idx"]
                trial_idx = task["trial_idx"]
                row = dict(task)
                row["sample_idx"] = sample_idx
                row["row_id"] = f"{problem_idx}_t{trial_idx}_s{sample_idx}"
                expanded.append(row)
    else:
        for task in tasks:
            problem_idx = task["problem_idx"]
            trial_idx = task["trial_idx"]
            for sample_idx in range(n_samples_per_trial):
                row = dict(task)
                row["sample_idx"] = sample_idx
                row["row_id"] = f"{problem_idx}_t{trial_idx}_s{sample_idx}"
                expanded.append(row)
    return expanded
