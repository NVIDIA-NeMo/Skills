# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
Harvest library docstrings for LibTrace.

Inputs: libraries available in the sandbox image.
Outputs: JSONL files under <output_dir> (per-library + unified).

If libraries are missing, rebuild the sandbox image with your deps.

Example:
  ns run_cmd --cluster=local --container=sandbox \
    --log_dir /workspace/libtrace-results/harvest-docs-chem/logs \
    "python /nemo_run/code/recipes/libtrace/scripts/harvest_docs.py \
      --domain chem --output_dir /workspace/libtrace-results/harvest-docs-chem/results"
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json as _json_std
from pathlib import Path

try:  # orjson is significantly faster; fallback to std json
    import orjson as _orjson  # type: ignore

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")


except Exception:  # pragma: no cover - best effort
    _orjson = None

    def _json_dumps(obj) -> str:
        return _json_std.dumps(obj, ensure_ascii=False)


def safe_getmembers(obj) -> list[tuple[str, object]]:
    try:
        names = dir(obj)
    except Exception:
        return []

    members: list[tuple[str, object]] = []
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        members.append((name, value))
    return members


PHYSICS_LIBRARIES = [
    "sympy",
    "numpy",
    "scipy",
    "pymatgen",
    "ase",
    "lammps",
    "qutip",
    "qiskit",
    "geant4_pybind",
    "uproot",
    "astropy",
    "sdss_access",
    "pymatgen.core",
]

CHEMISTRY_LIBRARIES = [
    "sympy",
    "numpy",
    "scipy",
    # Chemistry core
    "chempy",
    "mendeleev",
    "pubchempy",
    "thermo",
    "openbabel.openbabel",  # not "openbabel" (empty)
    # RDKit
    "rdkit.Chem",
    "rdkit.DataStructs",
    # Quantum chemistry - PySCF
    "pyscf",
    "pyscf.gto",
    "pyscf.scf",
    "pyscf.dft",
    "pyscf.mcscf",
    "pyscf.lib",
    "pyscf.symm",
    "pyscf.tddft",
    "pyscf.fci",
    # Atomic simulation - ASE
    "ase",
    "ase.build",
    "ase.atoms",
    "ase.constraints",
    "ase.lattice",
    "ase.geometry",
    "ase.optimize",
    "ase.utils",
    "ase.neighborlist",
    # Materials science - pymatgen
    "pymatgen.core",
    "pymatgen.core.structure",
    "pymatgen.core.periodic_table",
    "pymatgen.core.composition",
    "pymatgen.core.lattice",
    "pymatgen.core.surface",
    # Nuclear/radioactive
    "radioactivedecay",
    "radioactivedecay.inventory",
]

BIOLOGY_LIBRARIES = [
    "sympy",
    "numpy",
    "scipy",
    # Biopython - core and key submodules
    "Bio",
    "Bio.Seq",
    "Bio.SeqUtils",
    "Bio.SeqIO",
    "Bio.SeqRecord",
    "Bio.SeqFeature",
    "Bio.Align",
    "Bio.Phylo",
    "Bio.PDB",
    "Bio.Blast",
    "Bio.Entrez",
    "Bio.GenBank",
    "Bio.Restriction",  # 1091+ items - restriction enzymes
    "Bio.Cluster",
    "Bio.SearchIO",
    "Bio.SCOP",  # smaller but useful
    # Other bio libraries
    "skbio",
    "biotite",
    "dendropy",
    "ete3",
    "cobra",
    "bioservices",
    "pysam",
    "cyvcf2",
]

LIBRARY_GROUPS = {
    "physics": PHYSICS_LIBRARIES,
    "chem": CHEMISTRY_LIBRARIES,
    "bio": BIOLOGY_LIBRARIES,
}


def is_function_or_method(obj) -> bool:
    return inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj) or inspect.isroutine(obj)


def extract_docs_from_module(module, module_name: str, seen: set[str]) -> list[dict]:
    if module.__name__ in seen:
        return []
    seen.add(module.__name__)

    results: list[dict] = []
    for name, obj in safe_getmembers(module):
        if name.startswith("_"):
            continue

        full_name = f"{module_name}.{name}" if module_name else name
        if full_name == "scipy.LowLevelCallable":
            continue

        if is_function_or_method(obj):
            results.append(
                {
                    "name": full_name,
                    "type": "function",
                    "doc": inspect.getdoc(obj) or "",
                }
            )
            continue

        if inspect.isclass(obj):
            results.append(
                {
                    "name": full_name,
                    "type": "class",
                    "doc": inspect.getdoc(obj) or "",
                }
            )
            for method_name, method in safe_getmembers(obj):
                if method_name.startswith("_"):
                    continue
                if is_function_or_method(method):
                    method_full_name = f"{full_name}.{method_name}"
                    results.append(
                        {
                            "name": method_full_name,
                            "type": "method",
                            "doc": inspect.getdoc(method) or "",
                        }
                    )
            continue

        if inspect.ismodule(obj) and obj.__name__.startswith(module.__name__):
            results.extend(extract_docs_from_module(obj, obj.__name__, seen))

    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps(row) + "\n")


def process_library(library_name: str, output_dir: Path) -> list[dict]:
    try:
        lib = importlib.import_module(library_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing and missing != library_name:
            detail = f"Library '{library_name}' failed to import because dependency '{missing}' is missing."
        else:
            detail = f"Library '{library_name}' not found."
        raise RuntimeError(
            f"{detail} Check your container image, install the library, "
            "or remove it from --libraries / the selected library group."
        ) from exc
    docs = extract_docs_from_module(lib, library_name, seen=set())

    output_file = output_dir / f"{library_name}.jsonl"
    write_jsonl(output_file, docs)

    functions = sum(1 for doc in docs if doc["type"] == "function")
    classes = sum(1 for doc in docs if doc["type"] == "class")
    methods = sum(1 for doc in docs if doc["type"] == "method")

    print(f"\nExtracted {len(docs)} items from {library_name}")
    print(f"Saved to: {output_file}")
    print("Statistics:")
    print(f"  Functions: {functions}")
    print(f"  Classes: {classes}")
    print(f"  Methods: {methods}")

    for doc in docs:
        doc["source"] = library_name
    return docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract library docs for training.")
    parser.add_argument(
        "--domain",
        type=str,
        choices=sorted(LIBRARY_GROUPS.keys()),
        default=None,
        help="Domain / library group to process (ignored if --libraries is provided).",
    )
    parser.add_argument(
        "--libraries",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of library module names to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base directory for outputs.",
    )
    parser.add_argument(
        "--unified_name",
        type=str,
        default=None,
        help="Filename for the unified JSONL output.",
    )
    args = parser.parse_args()

    if args.domain is None and args.libraries is None:
        raise ValueError("Provide --domain or --libraries.")
    if args.domain is not None and args.libraries is not None:
        raise ValueError("Use only one of --domain or --libraries.")

    return args


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.libraries is not None:
        libraries = args.libraries
        unified_name = args.unified_name or "custom_unified_docs.jsonl"
    else:
        libraries = LIBRARY_GROUPS[args.domain]
        unified_name = args.unified_name or f"{args.domain}_unified_docs.jsonl"

    print("\nStarting documentation extraction...")
    all_docs: list[dict] = []
    for idx, library in enumerate(libraries, start=1):
        print(f"\nProcessing library {idx}/{len(libraries)}: {library}")
        all_docs.extend(process_library(library, output_dir))

    if not all_docs:
        raise RuntimeError("No docs extracted; check libraries and environment.")

    unified_file = output_dir / unified_name
    write_jsonl(unified_file, all_docs)

    print("\n" + "=" * 60)
    print("UNIFIED FILE STATISTICS:")
    print(f"Total items: {len(all_docs)}")

    library_counts: dict[str, int] = {}
    for doc in all_docs:
        source = doc["source"]
        library_counts[source] = library_counts.get(source, 0) + 1

    print("\nItems per library:")
    for lib, count in sorted(library_counts.items()):
        print(f"  {lib}: {count}")

    type_counts: dict[str, int] = {}
    for doc in all_docs:
        doc_type = doc["type"]
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

    print("\nItems by type:")
    for doc_type, count in sorted(type_counts.items()):
        print(f"  {doc_type}: {count}")

    print(f"\nUnified file saved to: {unified_file}")
