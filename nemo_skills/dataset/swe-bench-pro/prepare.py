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

import argparse
from pathlib import Path

import datasets

# Convert language codes to the same format as swe-bench-multilingual.
# This enables correct language-specific prompting and ignoring compilation files in git patches.
LANGUAGE_MAP = {
    "js": "javascript",
    "ts": "typescript",
    "go": "go",
    "python": "python",
}

# The following instances' dockerfiles are based on Alpine Linux (uses musl, not glibc).
# They have to be run in a separate eval job where the host Nemo-Skills container is also based on Alpine.
# Dockerfile: https://github.com/NVIDIA-NeMo/Skills/tree/main/dockerfiles/swe-bench/Dockerfile.nemo-skills.alpine
# This script creates separate dataset files for Alpine and Ubuntu instances.
ALPINE_INSTANCE_IDS = [
    "instance_flipt-io__flipt-86906cbfc3a5d3629a583f98e6301142f5f14bdb-v6bea0cc3a6fc532d7da914314f2944fc1cd04dee",
    "instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07",
    "instance_future-architect__vuls-e049df50fa1eecdccc5348e27845b5c783ed7c76-v73dc95f6b90883d8a87e01e5e9bb6d3cc32add6d",
    "instance_future-architect__vuls-e1fab805afcfc92a2a615371d0ec1e667503c254-v264a82e2f4818e30f5a25e4da53b27ba119f62b5",
    "instance_future-architect__vuls-e4728e388120b311c4ed469e4f942e0347a2689b-v264a82e2f4818e30f5a25e4da53b27ba119f62b5",
    "instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08b182eca4e2b9a17-v264a82e2f4818e30f5a25e4da53b27ba119f62b5",
    "instance_gravitational__teleport-005dcb16bacc6a5d5890c4cd302ccfd4298e275d-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-02d1efb8560a1aa1c72cfb1c08edd8b84a9511b4-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-0ecf31de0e98b272a6a2610abe1bbedd379a38a3-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-1a77b7945a022ab86858029d30ac7ad0d5239d00-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-1b08e7d0dbe68fe530a0f08ad408ec198b7c53fc-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-24cafecd8721891092210afc55f6413ab46ca211-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-2b15263e49da5625922581569834eec4838a9257-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-2bb3bbbd8aff1164a2353381cb79e1dc93b90d28-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-2be514d3c33b0ae9188e11ac9975485c853d98bb-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-32bcd71591c234f0d8b091ec01f1f5cbfdc0f13c-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-37c3724d0d6637e959e39408ee351565d73afe71-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-3a5c1e26394df2cb4fb3f01147fb9979662972c5-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-3ff19cf7c41f396ae468797d3aeb61515517edc9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-46aa81b1ce96ebb4ebed2ae53fd78cd44a05da6c-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-47530e1fd8bfb84ec096ebcbbc29990f30829655-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-4e1c39639edf1ab494dd7562844c8b277b5cfa18-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-4f771403dc4177dc26ee0370f7332f3fe54bee0f-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-53814a2d600ccd74c1e9810a567563432b98386e-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-645afa051b65d137654fd0d2d878a700152b305a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-65438e6e44b6ce51458d09b7bb028a2797cfb0ea-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-73cc189b0e9636d418c4470ecce0d9af5dae2f02-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-7744f72c6eb631791434b648ba41083b5f6d2278-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-8302d467d160f869b77184e262adbe2fbc95d9ba-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-ad41b3c15414b28a6cec8c25424a19bfa7abd0e9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-af5e2517de7d18406b614e413aca61c319312171-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-b1bcd8b90c474a35bb11cc3ef4cc8941e1f8eab2-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-b8fbb2d1e90ffcde88ed5fe9920015c1be075788-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-baeb2697c4e4870c9850ff0cd5c7a2d08e1401c9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-bb562408da4adeae16e025be65e170959d1ec492-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-cb712e3f0b06dadc679f895daef8072cae400c26-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-d6ffe82aaf2af1057b69c61bf9df777f5ab5635a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-d873ea4fa67d3132eccba39213c1ca2f52064dcc-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-e6895d8934f6e484341034869901145fbc025e72-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-e6d86299a855687b21970504fbf06f52a8f80c74-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-eefac60a350930e5f295f94a2d55b94c1988c04e-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_navidrome__navidrome-66b74c81f115c78cb69910b0472eeb376750efc4",
    "instance_protonmail__webclients-01ea5214d11e0df8b7170d91bafd34f23cb0f2b1",
    "instance_protonmail__webclients-0200ce0fc1d4dbd35178c10d440a284c82ecc858",
    "instance_protonmail__webclients-08bb09914d0d37b0cd6376d4cab5b77728a43e7b",
    "instance_protonmail__webclients-09fcf0dbdb87fa4f4a27700800ee4a3caed8b413",
    "instance_protonmail__webclients-0d0267c4438cf378bda90bc85eed3a3615871ac4",
    "instance_protonmail__webclients-0ec14e36ceb01ba45602a563e12352af8171ed39",
    "instance_protonmail__webclients-1917e37f5d9941a3459ce4b0177e201e2d94a622",
    "instance_protonmail__webclients-2dce79ea4451ad88d6bfe94da22e7f2f988efa60",
    "instance_protonmail__webclients-2f66db85455f4b22a47ffd853738f679b439593c",
    "instance_protonmail__webclients-32ff10999a06455cb2147f6873d627456924ae13",
    "instance_protonmail__webclients-369fd37de29c14c690cb3b1c09a949189734026f",
    "instance_protonmail__webclients-3a6790f480309130b5d6332dce6c9d5ccca13ee3",
    "instance_protonmail__webclients-4817fe14e1356789c90165c2a53f6a043c2c5f83",
    "instance_protonmail__webclients-4feccbc9990980aee26ea29035f8f931d6089895",
    "instance_protonmail__webclients-51742625834d3bd0d10fe0c7e76b8739a59c6b9f",
    "instance_protonmail__webclients-5d2576632037d655c3b6a28e98cd157f7e9a5ce1",
    "instance_protonmail__webclients-5e815cfa518b223a088fa9bb232a5fc90ab15691",
    "instance_protonmail__webclients-5f0745dd6993bb1430a951c62a49807c6635cd77",
    "instance_protonmail__webclients-6e1873b06df6529a469599aa1d69d3b18f7d9d37",
    "instance_protonmail__webclients-6f8916fbadf1d1f4a26640f53b5cf7f55e8bedb7",
    "instance_protonmail__webclients-708ed4a299711f0fa79a907cc5847cfd39c0fc71",
    "instance_protonmail__webclients-715dbd4e6999499cd2a576a532d8214f75189116",
    "instance_protonmail__webclients-7b833df125859e5eb98a826e5b83efe0f93a347b",
    "instance_protonmail__webclients-8142704f447df6e108d53cab25451c8a94976b92",
    "instance_protonmail__webclients-815695401137dac2975400fc610149a16db8214b",
    "instance_protonmail__webclients-8afd9ce04c8dde9e150e1c2b50d32e7ee2efa3e7",
    "instance_protonmail__webclients-8be4f6cb9380fcd2e67bcb18cef931ae0d4b869c",
    "instance_protonmail__webclients-944adbfe06644be0789f59b78395bdd8567d8547",
    "instance_protonmail__webclients-a6e6f617026794e7b505d649d2a7a9cdf17658c8",
    "instance_protonmail__webclients-ae36cb23a1682dcfd69587c1b311ae0227e28f39",
    "instance_protonmail__webclients-bf2e89c0c488ae1a87d503e5b09fe9dd2f2a635f",
    "instance_protonmail__webclients-c5a2089ca2bfe9aa1d85a664b8ad87ef843a1c9c",
    "instance_protonmail__webclients-c6f65d205c401350a226bb005f42fac1754b0b5b",
    "instance_protonmail__webclients-caf10ba9ab2677761c88522d1ba8ad025779c492",
    "instance_protonmail__webclients-cba6ebbd0707caa524ffee51c62b197f6122c902",
    "instance_protonmail__webclients-cfd7571485186049c10c822f214d474f1edde8d1",
    "instance_protonmail__webclients-d494a66038112b239a381f49b3914caf8d2ef3b4",
    "instance_protonmail__webclients-da91f084c0f532d9cc8ca385a701274d598057b8",
    "instance_protonmail__webclients-df60460f163fd5c34e844ab9015e3176f1ab1ac0",
    "instance_protonmail__webclients-dfe5604193d63bfcb91ce60d62db2f805c43bf11",
    "instance_protonmail__webclients-e65cc5f33719e02e1c378146fb981d27bc24bdf4",
    "instance_protonmail__webclients-e7f3f20c8ad86089967498632ace73c1157a9d51",
    "instance_protonmail__webclients-e9677f6c46d5ea7d277a4532a4bf90074f125f31",
    "instance_protonmail__webclients-f080ffc38e2ad7bddf2e93e5193e82c20c7a11e7",
    "instance_protonmail__webclients-f161c10cf7d31abf82e8d64d7a99c9fac5acfa18",
    "instance_protonmail__webclients-fc9d535e9beb3ae30a52a7146398cadfd6e30606",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_image}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time. "
        "See nemo_skills/dataset/swe-bench/dump_images.py",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ScaleAI/SWE-bench_Pro",
        help="Dataset name to load",
    )
    parser.add_argument("--split", type=str, default="test", help="Swe-Bench dataset split to use")
    parser.add_argument(
        "--setup",
        type=str,
        default="default",
        help="Setup name. Creates two dataset files: <setup>.alpine.jsonl and <setup>.ubuntu.jsonl",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter
    assert "{docker_image}" in container_formatter, "container_formatter must have {docker_image}"

    dataset = datasets.load_dataset(path=dataset_name, split=split)
    output_file_alpine = Path(__file__).parent / f"{args.setup}.alpine.jsonl"
    output_file_ubuntu = Path(__file__).parent / f"{args.setup}.ubuntu.jsonl"

    dataset = dataset.map(
        lambda x: {
            "language": LANGUAGE_MAP[x["repo_language"]],
            "problem_statement": (
                f"{x['problem_statement']}\n\n"
                f"Requirements:\n{x['requirements']}\n\n"
                f"New interfaces introduced:\n{x['interface']}"
            ),
        },
        remove_columns=["repo_language", "interface", "requirements"],
    )

    dataset = dataset.add_column(
        "container_formatter",
        [
            container_formatter.format(docker_image=f"jefzda/sweap-images:{row['dockerhub_tag']}")
            if container_formatter.startswith("docker://")
            else container_formatter.format(docker_image=f"jefzda_sweap-images_{row['dockerhub_tag']}")
            for row in dataset
        ],
    )
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset = dataset.add_column("container_repo_dir", ["/app"] * len(dataset))

    alpine_dataset = dataset.filter(lambda x: x["instance_id"] in ALPINE_INSTANCE_IDS)
    alpine_dataset.to_json(output_file_alpine, orient="records", lines=True)
    ubuntu_dataset = dataset.filter(lambda x: x["instance_id"] not in ALPINE_INSTANCE_IDS)
    ubuntu_dataset.to_json(output_file_ubuntu, orient="records", lines=True)
