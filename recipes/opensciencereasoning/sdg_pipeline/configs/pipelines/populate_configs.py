import yaml

# EOS: aops_openq  cdquestions_openq  scale  syn_gpqa_v1.2_4mcq  vedantu_physics_openq
# OCI: aops_mcq         olimpicos-aapt  syn_gpqa_v1.1_10mcq            vedantu_biology_chemistry_openq cdquestions_mcq  so              vedantu_biology_chemistry_mcq  vedantu_physics_mcq

CLUSTERS = ["eos-sandbox", "oci-sandbox"]
# CLUSTERS = ["oci-sandbox", "oci-sandbox"]
metadata = {
    "aops_openq": {
        "dataset_type": "openq",
        "dataset_name": "aops_openq",
        "cluster": CLUSTERS[0],
    },
    "olimpicos-aapt": {"dataset_type": "openq", "dataset_name": "olimpicos-aapt", "cluster": "oci-sandbox"},
    "vedantu_biology_chemistry_openq": {
        "dataset_type": "openq",
        "dataset_name": "vedantu_biology_chemistry_openq",
        "cluster": CLUSTERS[1],
    },
    "cdquestions_openq": {"dataset_type": "openq", "dataset_name": "cdquestions_openq", "cluster": "eos-sandbox"},
    "scale": {"dataset_type": "openq", "dataset_name": "scale", "cluster": "eos-sandbox"},
    "vedantu_physics_openq": {
        "dataset_type": "openq",
        "dataset_name": "vedantu_physics_openq",
        "cluster": CLUSTERS[0],
    },
    "aops_mcq": {"dataset_type": "mcq", "dataset_name": "aops_mcq", "cluster": CLUSTERS[1]},
    "vedantu_physics_mcq": {"dataset_type": "mcq", "dataset_name": "vedantu_physics_mcq", "cluster": CLUSTERS[1]},
    "cdquestions_mcq": {"dataset_type": "mcq", "dataset_name": "cdquestions_mcq", "cluster": CLUSTERS[1]},
    "syn_gpqa_v1.2_4mcq": {"dataset_type": "mcq", "dataset_name": "syn_gpqa_v1.2_4mcq", "cluster": CLUSTERS[0]},
    "so": {"dataset_type": "mcq", "dataset_name": "so", "cluster": CLUSTERS[1]},
    "vedantu_biology_chemistry_mcq": {
        "dataset_type": "mcq",
        "dataset_name": "vedantu_biology_chemistry_mcq",
        "cluster": CLUSTERS[1],
    },
    "syn_gpqa_v1.1_10mcq": {"dataset_type": "mcq10", "dataset_name": "syn_gpqa_v1.1_10mcq", "cluster": CLUSTERS[1]},
}


type2prompt = {
    "openq": "generic/search-boxed",
    "mcq": "eval/aai/search-mcq-4choices",
    "mcq10": "eval/aai/search-mcq-10choices",
}


with open("recipes/opensciencereasoning/configs/SDG_pipeline/gpt-conversion.yaml", "r") as f:
    gpt_partial_yaml = yaml.safe_load(f)

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

    # write to disk
    with open(
        f"recipes/opensciencereasoning/configs/SDG_pipeline/gpt-conversion-{key}.yaml",
        "w",
    ) as f:
        yaml.dump(gpt_partial_yaml, f)
    print(
        f"python recipes/opensciencereasoning/sdg_pipeline/pipeline/sdg_pipeline.py --stages bucket-qwen --config recipes/opensciencereasoning/configs/SDG_pipeline/gpt-conversion-{key}.yaml"
    )
    # convert_to_qwen_from_messages,remove_unused_fields,
    # convert_to_messages_format,convert_to_qwen_from_messages,remove_unused_fields,
