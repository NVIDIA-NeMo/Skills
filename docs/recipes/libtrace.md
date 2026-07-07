# LibTrace

LibTrace is a recipe for building domain-specific reasoning data from library
APIs. It harvests docstrings, labels applicability+relevance, generates
problems, solves them with a boxed-answer prompt, and gathers solutions.

The same workflow applies to chemistry, physics, and biology—swap the domain
inputs and output paths.

For the full walkthrough, command examples, and configuration details see
the [recipes/libtrace/README.md](https://github.com/NVIDIA-NeMo/Skills/blob/main/recipes/libtrace/README.md)
in the repository.

## Pipeline overview

1. **Harvest library docs** — extract public API docstrings from the sandbox container
2. **Prepare inference JSONL** — convert docs into an LLM-ready input file
3. **Label applicability + relevance** — LLM classifies each doc entry
4. **Filter** — keep only applicable, high-relevance entries
5. **Generate domain problems** — LLM creates problems based on filtered docs
6. **Collect generated problems** — merge and deduplicate across seeds
7. **Solve problems** — LLM solves with `generic/general-boxed` prompt and sandbox code execution
8. **Gather solutions** — compute stats and sample for training

## Files

All LibTrace scripts live in `recipes/libtrace/scripts/`.
Prompt templates are in `recipes/libtrace/prompts/`.
