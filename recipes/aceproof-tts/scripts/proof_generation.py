import os
import sys

from omegaconf import OmegaConf

PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from utils import (  # noqa: E402
    extract_boxed_answers,
    extract_self_eval,
    extract_solution,
    is_complete_finish_reason,
    load_system_prompt,
    response_metadata,
    strip_think,
)


def _parse_self_eval(text):
    try:
        self_eval_text = extract_self_eval(text).strip()
        solution_text = extract_solution(text).strip()
    except Exception:
        return text.strip(), {"self_eval": "null", "self_eval_score": 0}

    score = 0.0
    try:
        scores = [s.strip() for s in extract_boxed_answers(self_eval_text) if s.strip()]
        if scores:
            score = float(scores[-1])
    except Exception:
        score = 0.0

    return solution_text, {"self_eval": self_eval_text, "self_eval_score": score}


async def process_single(
    llm,
    datapoint,
    prompt_config_path,
    llm_kwargs,
    random_seed,
    system_prompt=None,
    system_prompt_path=None,
):
    prompt_template = OmegaConf.load(prompt_config_path).user
    prompt = prompt_template.format(question=datapoint["question"])

    system_prompt = load_system_prompt(system_prompt, system_prompt_path)
    if system_prompt is None:
        system_prompt = datapoint.get("system_prompt")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    seed = datapoint.get("generation_seed")
    if seed is None:
        seed = random_seed + datapoint.get("_async_position", 0)

    response = await llm.generate_async(prompt=messages, **llm_kwargs, random_seed=seed)
    generation = response.get("generation", "")
    finish_reason = response.get("finish_reason")

    generation_complete = is_complete_finish_reason(finish_reason)
    if generation_complete:
        proof_text = strip_think(generation)
        proof_text, self_eval = _parse_self_eval(proof_text)
        valid = bool(proof_text)
    else:
        proof_text = "UNFINISHED PROOF GENERATION"
        self_eval = {"self_eval": "null", "self_eval_score": 0}
        valid = False

    return {
        **datapoint,
        **response_metadata(response, "proof_generation"),
        "system_prompt": system_prompt,
        "prompt": prompt,
        "messages": messages,
        "generation": generation,
        "proof": proof_text,
        "self_eval": self_eval,
        "self_eval_score": self_eval.get("self_eval_score", 0),
        "finish_reason": finish_reason,
        "generation_complete": generation_complete,
        "valid": valid,
    }
