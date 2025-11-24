import yaml

CLUSTERS = ["eos-sandbox", "oci-sandbox"]

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
    "openq": "",
    "mcq": "recipes/opensciencereasoning/sdg_pipeline/configs/settings/mcq_4_options.yaml",
    "mcq10": "recipes/opensciencereasoning/sdg_pipeline/configs/settings/mcq_10_options.yaml",
}


with open("recipes/opensciencereasoning/sdg_pipeline/configs/pipelines/base.yaml", "r") as f:
    base_yaml = yaml.safe_load(f)

for key, value in metadata.items():
    # IO
    with open("recipes/opensciencereasoning/sdg_pipeline/configs/settings/info_input_output.yaml", "r") as f:
        info_input_output_config = yaml.safe_load(f)

    # Replace template variables in paths
    dataset_name = value["dataset_name"]
    dataset_type = value["dataset_type"]

    base_output_dir = info_input_output_config["base_output_dir"].replace("${dataset_name}", dataset_name)
    input_file = (
        info_input_output_config["input_file"]
        .replace("${dataset_type}", dataset_type)
        .replace("${dataset_name}", dataset_name)
    )

    base_yaml["base_output_dir"] = base_output_dir
    base_yaml["input_file"] = input_file
    base_yaml["dataset_name"] = dataset_name  # dataset name
    base_yaml["cluster"] = value["cluster"]  # cluster assignment

    # GPT OSS generation solution config
    with open("recipes/opensciencereasoning/sdg_pipeline/configs/settings/gen_sol_gpt_oss.yaml", "r") as f:
        gen_sol_gpt_oss_config = yaml.safe_load(f)
    base_yaml["stages"]["generate_solutions"] = gen_sol_gpt_oss_config["stages"]["generate_solutions"]

    # Question type config
    yaml_config = type2prompt[value["dataset_type"]]
    # if yaml_config, load it from the file
    if yaml_config:
        with open(yaml_config, "r") as f:
            question_type_config = yaml.safe_load(f)
        # Override the stages.generate_solutions section with the yaml_config
        base_yaml["stages"]["generate_solutions"] = question_type_config["stages"]["generate_solutions"]

    # write to disk
    with open(
        f"recipes/opensciencereasoning/sdg_pipeline/configs/pipelines/gpt-oss-{key}.yaml",
        "w",
    ) as f:
        yaml.dump(base_yaml, f)
    print(
        f"python recipes/opensciencereasoning/sdg_pipeline/pipeline/sdg_pipeline.py --config recipes/opensciencereasoning/sdg_pipeline/configs/pipelines/gpt-oss-{key}.yaml"
    )
