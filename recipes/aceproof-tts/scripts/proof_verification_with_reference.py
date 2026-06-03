import os
import sys

from omegaconf import OmegaConf

PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pipeline"))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from utils import (  # noqa: E402
    is_complete_finish_reason,
    load_system_prompt,
    parse_verification_score,
    response_metadata,
    strip_think,
)


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
    problem = datapoint.get("problem") or datapoint.get("question", "")
    reference_solution = datapoint.get("reference_solution", "")
    response = datapoint.get("response") or datapoint.get("proof", "")
    prompt = prompt_template.format(
        problem=problem,
        reference_solution=reference_solution,
        response=response,
    )

    system_prompt = load_system_prompt(system_prompt, system_prompt_path)
    if system_prompt is None:
        system_prompt = datapoint.get("system_prompt")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    seed = datapoint.get("verification_seed")
    if seed is None:
        seed = random_seed + datapoint.get("_async_position", 0)

    response = await llm.generate_async(prompt=messages, **llm_kwargs, random_seed=seed)
    generation = response.get("generation", "")
    finish_reason = response.get("finish_reason")

    rating_text = strip_think(generation)
    verification_complete = is_complete_finish_reason(finish_reason)
    score = parse_verification_score(rating_text) if verification_complete else None

    return {
        **datapoint,
        **response_metadata(response, "verification"),
        "system_prompt": system_prompt,
        "prompt": prompt,
        "messages": messages,
        "generation": generation,
        "rating_text": rating_text,
        "verification_score": score,
        "finish_reason": finish_reason,
        "verification_complete": verification_complete,
        "valid": score is not None and verification_complete,
    }
