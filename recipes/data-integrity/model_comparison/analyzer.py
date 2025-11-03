# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main analyzer class for model comparison"""

import logging
import os
import warnings
from datetime import datetime

import nltk
import spacy
from sentence_transformers import SentenceTransformer

from .analyses import (
    analyze_input_response_mapping_umap,
    analyze_multimodal_space_umap,
    analyze_response_embeddings_umap,
    analyze_response_lengths,
    analyze_semantic_similarity,
    analyze_vocabulary_diversity,
)
from .data_loader import load_and_prepare_data
from .report_generator import generate_analysis_report, generate_index_file
from .utils.file_utils import save_data
from .visualization.interactive_plots import create_interactive_explorer

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class OrganizedModelAnalyzer:
    """Enhanced analyzer with organized output structure and UMAP visualizations"""

    def __init__(self, json_file_path, results_base_dir="model_comparison_results"):
        """Initialize with path to JSON data file and results directory"""
        self.json_file_path = json_file_path
        self.results_base_dir = results_base_dir
        self.data = None
        self.df = None
        self.sentence_model = None
        self.nlp = None

        # Create results directory structure
        self.setup_results_directory()

    def setup_results_directory(self):
        """Create organized results directory structure"""
        # Create timestamp for this analysis run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract base filename for identification
        base_filename = os.path.splitext(os.path.basename(self.json_file_path))[0]

        # Create main results directory
        self.results_dir = os.path.join(self.results_base_dir, f"{base_filename}_{timestamp}")

        # Create subdirectories
        self.subdirs = {
            "visualizations": os.path.join(self.results_dir, "visualizations"),
            "data_outputs": os.path.join(self.results_dir, "data_outputs"),
            "reports": os.path.join(self.results_dir, "reports"),
        }

        # Create all directories
        os.makedirs(self.results_dir, exist_ok=True)
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)

        logger.info(f"📁 Created results directory: {self.results_dir}")

    def load_data(self):
        """Load and parse the JSON data"""
        self.data, self.df, summary_info = load_and_prepare_data(self.json_file_path)

        # Save loaded data summary
        save_data(self.subdirs, self.df, summary_info, "data_summary", "json")

    def initialize_models(self):
        """Initialize NLP models"""
        logger.info("🔧 Initializing NLP models...")
        try:
            self.sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.info(f"⚠️  Warning: Could not load SentenceTransformer model: {e}")

        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded")
        except Exception as e:
            logger.info(f"⚠️  Warning: Could not load spaCy model: {e}")

    def generate_final_report(self):
        """Generate final comprehensive report with UMAP analyses"""
        logger.info(f"\n{'🎯' * 20}")
        logger.info("GENERATING COMPREHENSIVE REPORT WITH UMAP ANALYSIS")
        logger.info(f"\n{'🎯' * 20}")

        # Run all analyses
        dt1 = datetime.now()
        length_stats, self.df = analyze_response_lengths(self.df, self.subdirs)
        dt2 = datetime.now()
        diversity_stats = analyze_vocabulary_diversity(self.df, self.subdirs)
        dt3 = datetime.now()
        _ = analyze_semantic_similarity(self.df, self.subdirs, self.sentence_model)
        dt4 = datetime.now()

        # Run UMAP analyses
        _ = analyze_response_embeddings_umap(self.df, self.subdirs, self.sentence_model)
        dt5 = datetime.now()
        _ = analyze_input_response_mapping_umap(self.df, self.subdirs, self.sentence_model)
        dt6 = datetime.now()
        _ = analyze_multimodal_space_umap(self.df, self.subdirs, self.sentence_model)
        dt7 = datetime.now()
        _ = create_interactive_explorer(self.df, self.subdirs, self.sentence_model)
        dt8 = datetime.now()

        # Calculate elapsed times
        elapsed_times = {
            "response_lengths": (dt2 - dt1).total_seconds() / 60,
            "vocabulary_diversity": (dt3 - dt2).total_seconds() / 60,
            "semantic_similarity": (dt4 - dt3).total_seconds() / 60,
            "response_embeddings_umap": (dt5 - dt4).total_seconds() / 60,
            "input_response_mapping_umap": (dt6 - dt5).total_seconds() / 60,
            "multimodal_space_umap": (dt7 - dt6).total_seconds() / 60,
            "interactive_umap_explorer": (dt8 - dt7).total_seconds() / 60,
        }

        # Log elapsed times
        for analysis, time in elapsed_times.items():
            logger.info(f"Elapsed time for {analysis}: {time:.2f} minutes")

        # Generate summary report
        generate_analysis_report(self.df, self.results_dir, self.subdirs, length_stats, diversity_stats, elapsed_times)

        # Create index file for easy navigation
        generate_index_file(self.results_dir, self.subdirs, self.df)

        logger.info("\n✅ Analysis Complete!")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        logger.info(
            f"📊 Total files generated: {len(os.listdir(self.subdirs['visualizations'])) + len(os.listdir(self.subdirs['data_outputs']))}"
        )
        logger.info(f"🔗 Open {os.path.join(self.results_dir, 'README.md')} for navigation guide")
        logger.info("🎮 All UMAP visualizations are interactive! Open the HTML files in your browser.")
        logger.info("🌟 5 interactive UMAP visualizations + dual similarity analysis (semantic + ROUGE-L) available")
