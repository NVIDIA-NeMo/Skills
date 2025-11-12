import yaml

# for data in [
#     "aops_openq",
#     "olimpicos-aapt",
#     "vedantu_biology_chemistry_openq",
#     "cdquestions_openq",
#     "scale",
#     "vedantu_physics_openq",
# ]:
# EOS: aops_openq  cdquestions_openq  scale  syn_gpqa_v1.2_4mcq  vedantu_physics_openq
# OCI: aops_mcq         olimpicos-aapt  syn_gpqa_v1.1_10mcq            vedantu_biology_chemistry_openq cdquestions_mcq  so              vedantu_biology_chemistry_mcq  vedantu_physics_mcq

metadata = {
    "aops_openq": {
        "dataset_type": "openq",
        "dataset_name": "aops_openq",
        "cluster": "eos-sandbox",
    },
    "olimpicos-aapt": {"dataset_type": "openq", "dataset_name": "olimpicos-aapt", "cluster": "oci-sandbox"},
    "vedantu_biology_chemistry_openq": {
        "dataset_type": "openq",
        "dataset_name": "vedantu_biology_chemistry_openq",
        "cluster": "oci-sandbox",
    },
    "cdquestions_openq": {"dataset_type": "openq", "dataset_name": "cdquestions_openq", "cluster": "eos-sandbox"},
    "scale": {"dataset_type": "openq", "dataset_name": "scale", "cluster": "eos-sandbox"},
    "vedantu_physics_openq": {
        "dataset_type": "openq",
        "dataset_name": "vedantu_physics_openq",
        "cluster": "eos-sandbox",
    },
    "aops_mcq": {"dataset_type": "mcq", "dataset_name": "aops_mcq", "cluster": "oci-sandbox"},
    "vedantu_physics_mcq": {"dataset_type": "mcq", "dataset_name": "vedantu_physics_mcq", "cluster": "oci-sandbox"},
    "cdquestions_mcq": {"dataset_type": "mcq", "dataset_name": "cdquestions_mcq", "cluster": "oci-sandbox"},
    "syn_gpqa_v1.2_4mcq": {"dataset_type": "mcq", "dataset_name": "syn_gpqa_v1.2_4mcq", "cluster": "eos-sandbox"},
    "so": {"dataset_type": "mcq", "dataset_name": "so", "cluster": "oci-sandbox"},
    "vedantu_biology_chemistry_mcq": {
        "dataset_type": "mcq",
        "dataset_name": "vedantu_biology_chemistry_mcq",
        "cluster": "oci-sandbox",
    },
    "syn_gpqa_v1.1_10mcq": {"dataset_type": "mcq10", "dataset_name": "syn_gpqa_v1.1_10mcq", "cluster": "oci-sandbox"},
}


type2prompt = {
    "openq": "generic/search-boxed",
    "mcq": "eval/aai/search-mcq-4choices",
    "mcq10": "eval/aai/search-mcq-10choices",
}


with open("recipes/opensciencereasoning/configs/SDG_pipeline/gpt-partial.yaml", "r") as f:
    gpt_partial_yaml = yaml.safe_load(f)
print(gpt_partial_yaml)
for key, value in metadata.items():
    prompt = type2prompt[value["dataset_type"]]
    cluster = value["cluster"]
    gpt_partial_yaml["dataset_name"] = value["dataset_name"]
    gpt_partial_yaml["dataset_type"] = value["dataset_type"]
    gpt_partial_yaml["prompt"] = prompt
    gpt_partial_yaml["cluster"] = cluster
    if value["dataset_type"] == "openq":
        gpt_partial_yaml["stages"]["generate_solutions"]["predicted_answer_regex"] = None
    else:
        gpt_partial_yaml["stages"]["generate_solutions"]["predicted_answer_regex"] = "Answer:\s*([A-J])\s*$"
    gpt_partial_yaml["stages"]["generate_solutions"]["end_reasoning_string"] = (
        "'++end_reasoning_string=\"<|start|>assistant<|channel|>final<|message|>\"'"
    )

    gpt_partial_yaml["stages"]["generate_solutions"]["generation_kwargs"]["args"]["num_chunks"] = 20

    # write to disk
    with open(
        f"recipes/opensciencereasoning/configs/SDG_pipeline/gpt-partial-{key}.yaml",
        "w",
    ) as f:
        yaml.dump(gpt_partial_yaml, f)
