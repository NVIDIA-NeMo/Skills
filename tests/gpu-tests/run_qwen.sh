# will run all tests starting from only a HF checkpoint. Only requires 1 GPU.
# also need to define HF_TOKEN for some of the tests
set -e

export NEMO_SKILLS_TEST_MODEL_TYPE=qwen

# Switch to Qwen3 model for other tests
export NEMO_SKILLS_TEST_HF_MODEL=Qwen/Qwen3-1.7B
# generation/evaluation tests
pytest tests/gpu-tests/test_eval.py -s -x
pytest tests/gpu-tests/test_generate.py -s -x
pytest tests/gpu-tests/test_judge.py -s -x
pytest tests/gpu-tests/test_run_cmd_llm_infer.py -s -x
pytest tests/gpu-tests/test_nemo_evaluator.py -s -x

# For contamination test, reasoning models are not a good choice. Switching to a instruct model.
export NEMO_SKILLS_TEST_HF_MODEL=Qwen/Qwen3-4B-Instruct-2507
pytest tests/gpu-tests/test_contamination.py -s -x

# TODO: Add fast context retry tests
# pytest tests/gpu-tests/test_context_retry.py -s -x

# for sft we are using the tiny random model to run much faster
ns run_cmd --cluster test-local --config_dir tests/gpu-tests --container vllm \
    python3 /nemo_run/code/tests/gpu-tests/make_tiny_llm.py --model_type $NEMO_SKILLS_TEST_MODEL_TYPE

# converting the model through test
export NEMO_SKILLS_TEST_HF_MODEL=/tmp/nemo-skills-tests/$NEMO_SKILLS_TEST_MODEL_TYPE/tiny-model-hf
# training tests
pytest tests/gpu-tests/test_train.py -s -x
