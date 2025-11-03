# Data Integrity Scripts - Model Comparison and Analysis

A comprehensive toolkit for preparing data from the NVIDIA Llama-Nemotron Post-Training Dataset and performing detailed comparative analysis of language model outputs using various analytical techniques.

## 📋 Overview

This project consists of two main components:
1. **Data Preparation Scripts** - Download and prepare data from NVIDIA's dataset
2. **Model Comparison Module** - Analyze and compare model outputs using UMAP visualizations, similarity analysis, vocabulary metrics, and more

## 🚀 Quick Start

### 1. Data Preparation

```bash
# Download data with existing responses
python prepare_data.py --split science --nrows 1000 --generator DeepSeek-R1
```

Parameters:
- `--split`: Dataset split ('safety', 'chat', 'science', 'code', 'math')
- `--nrows`: Number of examples to download (default: 100)
- `--generator`: Model whose responses to compare (default: DeepSeek-R1)

See `download` in `run_integrity_pipeline.py`

### 2. Generate alternative response from a different model

- Invoke NeMo-Skill to generate alternative response from target model
- Result will be saved in `answers_dir` directory
- see `gen_answer` in `run_integrity_pipeline.py`

### 3. Postprocess the result to create a dataset with required fields

```bash
python postprocess_data.py --input_path answers_dir --target_model target_model --output_path result.json
```
See `postprocess` in `run_integrity_pipeline.py`


### 4. Model Comparison Analysis

```bash
# Run analysis on prepared data
python -m model_comparison.main --json_file result.json --result_dir ./analysis_results
```
See `compare` in `run_integrity_pipeline.py`


## 📁 Project Structure

```
scripts/
├── README.md                 # This file
├── prepare_data.py           # Download from HuggingFace nvidia/Llama-Nemotron-Post-Training-Dataset
├── postprocess_data.py       # After generate answers from NeMo-Skill, prepare the result for comparison
├── run_integrity_pipeline.py # A pipeline script to run end-to-end from dowloading to comparison
│
└── model_comparison/         # Analysis module
    ├── __init__.py
    ├── analyzer.py           # Main OrganizedModelAnalyzer class
    ├── data_loader.py        # Data loading and preprocessing
    ├── main.py               # Command-line entry point
    ├── report_generator.py   # Report generation utilities
    │
    ├── analyses/             # Analysis modules
    │   ├── length_analysis.py
    │   ├── vocabulary_analysis.py
    │   ├── similarity_analysis.py
    │   └── umap_analysis.py
    │
    ├── visualization/        # Visualization modules
    │   ├── static_plots.py
    │   └── interactive_plots.py
    │
    └── utils/                # Utility functions
        ├── text_utils.py
        ├── model_utils.py
        └── file_utils.py
```

## 🔧 Installation

### Requirements

1. **Basic Requirements**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk spacy \
    sentence-transformers umap-learn plotly textstat textblob rouge \
    datasets tqdm openai
```

2. **Additional Setup**
```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm
```


## 📈 Model Comparison Analyses

### 1. Response Length Analysis
- Character, word, and sentence count statistics
- Distribution visualizations
- Comparative box plots

### 2. Vocabulary Diversity Analysis
- Type-Token Ratio (TTR)
- Hapax legomena analysis
- Vocabulary richness metrics
- Most common words

### 3. Semantic Similarity Analysis
- Cosine similarity using sentence embeddings
- ROUGE-L scores for lexical overlap
- Pairwise comparison matrices
- Distribution histograms

### 4. UMAP Visualizations (Interactive)
- **Response Embeddings**: Semantic landscape of all responses
- **Input-Response Mapping**: Relationship between prompts and outputs
- **Multimodal Space**: Combined semantic and stylistic features
- **Interactive Explorer**: Detailed response exploration

## 📋 Expected Output

```
analysis_results/
├── visualizations/          # All plots and interactive HTML files
│   ├── *_response_length_analysis.png
│   ├── *_vocabulary_diversity_analysis.png
│   ├── *_similarity_comparison_heatmaps.png
│   ├── *_response_embeddings_umap.html
│   └── ...
│
├── data_outputs/           # CSV and JSON data files
│   ├── *_length_statistics.csv
│   ├── *_vocabulary_diversity_metrics.csv
│   ├── *_semantic_similarity_detailed.csv
│   └── ...
│
└── reports/               # Summary reports
    └── *_analysis_report.md
```

## 💡 Usage Examples

### Complete Workflow

```bash
python run_integrity_pipeline.py
```

## 📊 Input Data Format to comparison script

The JSON file should have the following structure:
```json
[
  {
    "input": [{"content": "prompt text"}],
    "output_model1": "response from model 1",
    "output_model2": "response from model 2",
    "split": "train/test/val (optional)",
    "generator": "original model name",
    "system_prompt": "system instructions",
    "reasoning": "reasoning process (if available)"
  },
  ...
]
```

## 🎯 Key Features

1. **End-to-End Pipeline**: From data preparation to comprehensive analysis
2. **Modular Design**: Each analysis type is in its own module
3. **Interactive Visualizations**: UMAP plots are interactive HTML files
4. **Parallel Processing**: Efficient data preparation and analysis
5. **Flexible Input**: Supports various model output formats
6. **Comprehensive Metrics**: Both semantic and lexical analysis

## 📈 Interpreting Results

### Response Length
- **Longer responses**: More detailed/verbose models
- **Consistent lengths**: Controlled generation
- **High variance**: Context-dependent behavior

### Vocabulary Diversity
- **Higher TTR**: More diverse vocabulary
- **More hapax legomena**: Using unique words more often
- **Higher vocabulary richness**: Better lexical diversity

### Similarity Analysis
- **Semantic similarity > 0.8**: Very similar meanings
- **ROUGE-L > 0.6**: High lexical overlap
- **Low similarity**: Different approaches to same prompt

### UMAP Visualizations
- **Clusters**: Similar response types
- **Spread**: Diversity of responses
- **Model separation**: Distinct generation styles

## 🤝 Contributing

To add new features:
1. For data preparation: Modify prepare_*.py scripts
2. For new analysis types:
   - Create a new module in `model_comparison/analyses/`
   - Add visualization functions in `model_comparison/visualization/`
   - Update the analyzer to call your analysis
   - Update the report generator
