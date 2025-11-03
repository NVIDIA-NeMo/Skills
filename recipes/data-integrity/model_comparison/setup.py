#!/usr/bin/env python3
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

"""
Setup script for model_comparison module
Installs dependencies and downloads required data
"""

import subprocess
import sys


def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    import nltk

    nltk.download("punkt")
    nltk.download("stopwords")
    print("✅ NLTK data downloaded")


def download_spacy_model():
    """Download spaCy English model"""
    print("\n🌐 Downloading spaCy English model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    print("✅ spaCy model downloaded")


def verify_installation():
    """Verify that all modules can be imported"""
    print("\n🔍 Verifying installation...")

    modules_to_check = {
        "pandas": "Core data manipulation",
        "numpy": "Numerical operations",
        "matplotlib": "Static plotting",
        "seaborn": "Statistical visualizations",
        "sklearn": "Machine learning utilities",
        "nltk": "Natural language processing",
        "spacy": "Advanced NLP",
        "sentence_transformers": "Sentence embeddings",
        "textstat": "Text statistics",
        "textblob": "Text processing",
        "umap": "Dimensionality reduction (optional)",
        "plotly": "Interactive visualizations (optional)",
        "rouge": "ROUGE metrics (optional)",
    }

    failed_imports = []
    optional_missing = []

    for module, description in modules_to_check.items():
        try:
            __import__(module)
            print(f"✅ {module:<20} - {description}")
        except ImportError:
            if module in ["umap", "plotly", "rouge"]:
                optional_missing.append((module, description))
                print(f"⚠️  {module:<20} - {description} [OPTIONAL]")
            else:
                failed_imports.append((module, description))
                print(f"❌ {module:<20} - {description}")

    if failed_imports:
        print("\n❌ ERROR: Required modules failed to import:")
        for module, desc in failed_imports:
            print(f"   - {module}: {desc}")
        sys.exit(1)

    if optional_missing:
        print("\n⚠️  Optional modules not installed:")
        for module, desc in optional_missing:
            print(f"   - {module}: {desc}")
        print("\nThese are optional but recommended for full functionality.")

    print("\n✅ All required modules installed successfully!")


def main():
    """Run the complete setup process"""
    print("🚀 Setting up model_comparison module...\n")

    try:
        # Install packages
        install_requirements()

        # Download NLTK data
        download_nltk_data()

        # Download spaCy model
        download_spacy_model()

        # Verify installation
        verify_installation()

        print("\n🎉 Setup complete! You can now use the model_comparison module.")
        print("\nExample usage:")
        print("  python -m model_comparison.main --json_file /path/to/data.json --result_dir /path/to/results")

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
