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

# This is similar to nemo_skills/dataset/swe-bench/prepare.py, but adds an extra language column.

import argparse
from pathlib import Path

import datasets

# the following instances use Alpine Linux (uses musl, not glibc), as a result the setup command results in a failure
# Error relocating /root/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12: posix_fallocate64: symbol not found
# TODO: we need to find a way to handle this gracefully.
unsupported_instance_ids = [
    "instance_gravitational__teleport-8302d467d160f869b77184e262adbe2fbc95d9ba-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-7744f72c6eb631791434b648ba41083b5f6d2278-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-1b08e7d0dbe68fe530a0f08ad408ec198b7c53fc-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-406f9396ad65696d58865b3a6283109cd4eaf40e",
    "instance_flipt-io__flipt-492cc0b158200089dceede3b1aba0ed28df3fb1d",
    "instance_flipt-io__flipt-5af0757e96dec4962a076376d1bedc79de0d4249",
    "instance_flipt-io__flipt-29d3f9db40c83434d0e3cc082af8baec64c391a9",
    "instance_flipt-io__flipt-dae029cba7cdb98dfb1a6b416c00d324241e6063",
    "instance_future-architect__vuls-e4728e388120b311c4ed469e4f942e0347a2689b-v264a82e2f4818e30f5a25e4da53b27ba119f62b5",
    "instance_flipt-io__flipt-c12967bc73fdf02054cf3ef8498c05e25f0a18c0",
    "instance_flipt-io__flipt-f743945d599b178293e89e784b3b2374b1026430",
    "instance_flipt-io__flipt-e91615cf07966da41756017a7d571f9fc0fdbe80",
    "instance_flipt-io__flipt-5c7037ececb0bead0a8eb56054e224bcd7ac5922",
    "instance_flipt-io__flipt-8bd3604dc54b681f1f0f7dd52cbc70b3024184b6",
    "instance_flipt-io__flipt-84806a178447e766380cc66b14dee9c6eeb534f4",
    "instance_flipt-io__flipt-b2393f07d893024ab1e47ea2081e0289e1f9d56f",
    "instance_flipt-io__flipt-3ef34d1fff012140ba86ab3cafec8f9934b492be",
    "instance_flipt-io__flipt-40007b9d97e3862bcef8c20ae6c87b22ea0627f0",
    "instance_flipt-io__flipt-e2bd19dafa7166c96b082fb2a59eb54b4be0d778",
    "instance_flipt-io__flipt-b2cd6a6dd73ca91b519015fd5924fde8d17f3f06",
    "instance_flipt-io__flipt-02e21636c58e86c51119b63e0fb5ca7b813b07b1",
    "instance_flipt-io__flipt-e88e93990e3ec1e7697754b423decc510d5dd5fe",
    "instance_flipt-io__flipt-21a935ad7886cc50c46852be21b37f363a926af0",
    "instance_gravitational__teleport-24cafecd8721891092210afc55f6413ab46ca211-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-aebaecd026f752b187f11328b0d464761b15d2ab",
    "instance_flipt-io__flipt-cf06f4ebfab7fa21eed3e5838592e8e44566957f",
    "instance_flipt-io__flipt-72d06db14d58692bfb4d07b1aa745a37b35956f3",
    "instance_flipt-io__flipt-56a620b8fc9ef7a0819b47709aa541cdfdbba00b",
    "instance_gravitational__teleport-645afa051b65d137654fd0d2d878a700152b305a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_future-architect__vuls-e049df50fa1eecdccc5348e27845b5c783ed7c76-v73dc95f6b90883d8a87e01e5e9bb6d3cc32add6d",
    "instance_gravitational__teleport-b1bcd8b90c474a35bb11cc3ef4cc8941e1f8eab2-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-d873ea4fa67d3132eccba39213c1ca2f52064dcc-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_flipt-io__flipt-c188284ff0c094a4ee281afebebd849555ebee59",
    "instance_gravitational__teleport-d6ffe82aaf2af1057b69c61bf9df777f5ab5635a-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-b6cef5cdc0daff3ee99e5974ed60a1dc6b4b0d67",
    "instance_flipt-io__flipt-9d25c18b79bc7829a6fb08ec9e8793d5d17e2868",
    "instance_flipt-io__flipt-e594593dae52badf80ffd27878d2275c7f0b20e9",
    "instance_flipt-io__flipt-f36bd61fb1cee4669de1f00e59da462bfeae8765",
    "instance_flipt-io__flipt-f1bc91a1b999656dbdb2495ccb57bf2105b84920",
    "instance_flipt-io__flipt-d966559200183b713cdf3ea5007a7e0ba86a5afb",
    "instance_gravitational__teleport-1a77b7945a022ab86858029d30ac7ad0d5239d00-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-65438e6e44b6ce51458d09b7bb028a2797cfb0ea-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_flipt-io__flipt-2ce8a0331e8a8f63f2c1b555db8277ffe5aa2e63",
    "instance_flipt-io__flipt-5aef5a14890aa145c22d864a834694bae3a6f112",
    "instance_flipt-io__flipt-756f00f79ba8abf9fe53f3c6c818123b42eb7355",
    "instance_future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a-v1151a6325649aaf997cd541ebe533b53fddf1b07",
    "instance_flipt-io__flipt-7161f7b876773a911afdd804b281e52681cb7321",
    "instance_flipt-io__flipt-2eac0df47b5ecc8bb05002d80383ceb08ab3620a",
    "instance_flipt-io__flipt-3d5a345f94c2adc8a0eaa102c189c08ad4c0f8e8",
    "instance_flipt-io__flipt-86906cbfc3a5d3629a583f98e6301142f5f14bdb-v6bea0cc3a6fc532d7da914314f2944fc1cd04dee",
    "instance_flipt-io__flipt-e42da21a07a5ae35835ec54f74004ebd58713874",
    "instance_flipt-io__flipt-ee02b164f6728d3227c42671028c67a4afd36918",
    "instance_flipt-io__flipt-05d7234fa582df632f70a7cd10194d61bd7043b9",
    "instance_flipt-io__flipt-6fd0f9e2587f14ac1fdd1c229f0bcae0468c8daa",
    "instance_gravitational__teleport-53814a2d600ccd74c1e9810a567563432b98386e-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_protonmail__webclients-944adbfe06644be0789f59b78395bdd8567d8547",
    "instance_flipt-io__flipt-65581fef4aa807540cb933753d085feb0d7e736f",
    "instance_flipt-io__flipt-524f277313606f8cd29b299617d6565c01642e15",
    "instance_flipt-io__flipt-a42d38a1bb1df267c53d9d4a706cf34825ae3da9",
    "instance_gravitational__teleport-4f771403dc4177dc26ee0370f7332f3fe54bee0f-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-dbe263961b187e1c5d7fe34c65b000985a2da5a0",
    "instance_flipt-io__flipt-f808b4dd6e36b9dc8b011eb26b196f4e2cc64c41",
    "instance_flipt-io__flipt-0b119520afca1cf25c470ff4288c464d4510b944",
    "instance_flipt-io__flipt-c1728053367c753688f114ec26e703c8fdeda125",
    "instance_flipt-io__flipt-c1fd7a81ef9f23e742501bfb26d914eb683262aa",
    "instance_gravitational__teleport-af5e2517de7d18406b614e413aca61c319312171-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-c8d71ad7ea98d97546f01cce4ccb451dbcf37d3b",
    "instance_gravitational__teleport-32bcd71591c234f0d8b091ec01f1f5cbfdc0f13c-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-3ff19cf7c41f396ae468797d3aeb61515517edc9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-96820c3ad10b0b2305e8877b6b303f7fafdf815f",
    "instance_flipt-io__flipt-e50808c03e4b9d25a6a78af9c61a3b1616ea356b",
    "instance_flipt-io__flipt-cd18e54a0371fa222304742c6312e9ac37ea86c1",
    "instance_flipt-io__flipt-b22f5f02e40b225b6b93fff472914973422e97c6",
    "instance_flipt-io__flipt-292fdaca9be39e6a921aaa8874c011d0fdd3e874",
    "instance_gravitational__teleport-ad41b3c15414b28a6cec8c25424a19bfa7abd0e9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-b433bd05ce405837804693bebd5f4b88d87133c8",
    "instance_gravitational__teleport-eefac60a350930e5f295f94a2d55b94c1988c04e-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-5ffba3406a7993d97ced4cc13658bee66150fcca",
    "instance_flipt-io__flipt-b68b8960b8a08540d5198d78c665a7eb0bea4008",
    "instance_flipt-io__flipt-381b90f718435c4694380b5fcd0d5cf8e3b5a25a",
    "instance_flipt-io__flipt-3b2c25ee8a3ac247c3fad13ad8d64ace34ec8ee7",
    "instance_flipt-io__flipt-b3cd920bbb25e01fdb2dab66a5a913363bc62f6c",
    "instance_flipt-io__flipt-507170da0f7f4da330f6732bffdf11c4df7fc192",
    "instance_protonmail__webclients-815695401137dac2975400fc610149a16db8214b",
    "instance_flipt-io__flipt-2ca5dfb3513e4e786d2b037075617cccc286d5c3",
    "instance_future-architect__vuls-e1fab805afcfc92a2a615371d0ec1e667503c254-v264a82e2f4818e30f5a25e4da53b27ba119f62b5",
    "instance_flipt-io__flipt-6fe76d024ee0c50ddb09c86f4ae0bd4c208fd65f",
    "instance_flipt-io__flipt-af7a0be46d15f0b63f16a868d13f3b48a838e7ce",
    "instance_gravitational__teleport-2b15263e49da5625922581569834eec4838a9257-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-cd2f3b0a9d4d8b8a6d3d56afab65851ecdc408e8",
    "instance_future-architect__vuls-ef2be3d6ea4c0a13674aaab08b182eca4e2b9a17-v264a82e2f4818e30f5a25e4da53b27ba119f62b5",
    "instance_flipt-io__flipt-1737085488ecdcd3299c8e61af45a8976d457b7e",
    "instance_flipt-io__flipt-b2170346dc37cf42fda1386cd630f24821ad2ac5",
    "instance_gravitational__teleport-e6895d8934f6e484341034869901145fbc025e72-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_flipt-io__flipt-0fd09def402258834b9d6c0eaa6d3b4ab93b4446",
    "instance_gravitational__teleport-37c3724d0d6637e959e39408ee351565d73afe71-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-02d1efb8560a1aa1c72cfb1c08edd8b84a9511b4-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_flipt-io__flipt-db1c3b100e231c62f0c90c2ab037614f20a2a63b",
    "instance_gravitational__teleport-b8fbb2d1e90ffcde88ed5fe9920015c1be075788-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-9f8127f225a86245fa35dca4885c2daef824ee55",
    "instance_gravitational__teleport-46aa81b1ce96ebb4ebed2ae53fd78cd44a05da6c-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-47530e1fd8bfb84ec096ebcbbc29990f30829655-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-cb712e3f0b06dadc679f895daef8072cae400c26-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-1dceb5edf3fa8f39495b939ef9cc0c3dd38fa17d",
    "instance_gravitational__teleport-e6d86299a855687b21970504fbf06f52a8f80c74-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-0ecf31de0e98b272a6a2610abe1bbedd379a38a3-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-73cc189b0e9636d418c4470ecce0d9af5dae2f02-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-518ec324b66a07fdd95464a5e9ca5fe7681ad8f9",
    "instance_flipt-io__flipt-e5fe37c379e1eec2dd3492c5737c0be761050b26",
    "instance_flipt-io__flipt-b4bb5e13006a729bc0eed8fe6ea18cff54acdacb",
    "instance_gravitational__teleport-baeb2697c4e4870c9850ff0cd5c7a2d08e1401c9-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-967855b429f749c28c112b8cb1b15bc79157f973",
    "instance_gravitational__teleport-4e1c39639edf1ab494dd7562844c8b277b5cfa18-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_gravitational__teleport-2be514d3c33b0ae9188e11ac9975485c853d98bb-vce94f93ad1030e3136852817f2423c1b3ac37bc4",
    "instance_gravitational__teleport-bb562408da4adeae16e025be65e170959d1ec492-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-c6a7b1fd933e763b1675281b30077e161fa115a1",
    "instance_flipt-io__flipt-ea9a2663b176da329b3f574da2ce2a664fc5b4a1",
    "instance_flipt-io__flipt-a0cbc0cb65ae601270bdbe3f5313e2dfd49c80e4",
    "instance_gravitational__teleport-3a5c1e26394df2cb4fb3f01147fb9979662972c5-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-36e62baffae2132f78f9d34dc300a9baa2d7ae0e",
    "instance_gravitational__teleport-2bb3bbbd8aff1164a2353381cb79e1dc93b90d28-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-c154dd1a3590954dfd3b901555fc6267f646a289",
    "instance_flipt-io__flipt-15b76cada1ef29cfa56b0fba36754be36243dded",
    "instance_flipt-io__flipt-ebb3f84c74d61eee4d8c6875140b990eee62e146",
    "instance_gravitational__teleport-005dcb16bacc6a5d5890c4cd302ccfd4298e275d-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "instance_flipt-io__flipt-abaa5953795afb9c621605bb18cb32ac48b4508c",
    "instance_flipt-io__flipt-690672523398c2b6f6e4562f0bf9868664ab894f",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_tag}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time. "
        "See nemo_skills/dataset/swe-bench/dump_images.py",
    )
    parser.add_argument("--split", type=str, default="test", help="Swe-Bench dataset split to use")
    parser.add_argument(
        "--setup", type=str, default="default", help="Setup name (used as nemo-skills split parameter)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ScaleAI/SWE-bench_Pro",
        help="Dataset name to load",
    )
    parser.add_argument(
        "--exclude_unsupported_instance_ids",
        action="store_true",
        default=True,
        help="Exclude unsupported instance ids",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter
    assert "{docker_tag}" in container_formatter, "container_formatter must have {docker_tag}"

    dataset = datasets.load_dataset(path=dataset_name, split=split)
    output_file = Path(__file__).parent / f"{args.setup}.jsonl"

    dataset = dataset.rename_column("repo_language", "language")
    dataset = dataset.add_column(
        "container_formatter",
        [
            container_formatter.format(docker_tag=f"jefzda/sweap-images:{row['dockerhub_tag']}")
            if container_formatter.startswith("docker://")
            else container_formatter.format(docker_tag=f"jefzda_sweap-images_{row['dockerhub_tag']}")
            for row in dataset
        ],
    )
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    if args.exclude_unsupported_instance_ids:
        dataset = dataset.filter(lambda x: x["instance_id"] not in unsupported_instance_ids)
    dataset.to_json(output_file, orient="records", lines=True)
