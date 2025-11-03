# Reproducing the OpenScience Dataset Collection

This recipe contains the scripts and prompts to reproduce the **OpenScience** datasets:

* [OpenScienceReasoning-2](https://huggingface.co/datasets/nvidia/OpenScienceReasoning-2)
* [OpenScience](https://huggingface.co/datasets/nvidia/OpenScience)

## What We Share

This repository provides all the necessary components to run the data generation recipe:

* **Prompts for Data Augmentation**: Templates for generating "similar" and "inspired-by" questions to expand the dataset (for example, [`prompts/mcq_augment_similar.yaml`](prompts/mcq_augment_similar.yaml) and [`prompts/mcq_augment_inspired_by.yaml`](prompts/mcq_augment_inspired_by.yaml)).
* **Prompts for Question Generation**: Templates for creating multiple-choice questions (MCQs) with either four or ten answer options (see [`prompts/`](prompts/)).
* **Prompt for Subtopic Expansion**: The template used to generate a list of subtopics (e.g., [`prompts/SDG_pipeline/topics_labeling.yaml`](prompts/SDG_pipeline/topics_labeling.yaml)).
* **Solution Filtering Script**: A script to filter generated solutions based on majority voting, as discussed in our paper ([`scripts/SDG_pipeline/filter_solutions.py`](scripts/SDG_pipeline/filter_solutions.py)).
