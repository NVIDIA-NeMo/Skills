import argparse

from nemo_skills.pipeline.cli import wrap_arguments, eval

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmarks', type=str, required=True)
    parser.add_argument('-s', '--split', type=str, default='all')

    parser.add_argument('-e', '--exp-name', type=str, required=True)
    parser.add_argument('-c', '--cluster', type=str, required=True)

    parser.add_argument('-m', '--model-path', type=str, required=True)
    parser.add_argument('-t', '--temperature', type=float, default=1.0)
    parser.add_argument('-p', '--top-p', type=float, default=1.0)
    parser.add_argument('-k', '--top-k', type=int, default=-1)
    parser.add_argument('-l', '--max-model-len', type=int, default=8192)

    parser.add_argument('-g', '--num-gpus', type=int, default=8)
    parser.add_argument('-n', '--num-nodes', type=int, default=1)

    parser.add_argument('-o', '--output', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    eval(
        ctx=wrap_arguments(
            f"++inference.temperature={args.temperature} "
            f"++inference.top_p={args.top_p} "
            f"++inference.top_k={args.top_k} "
            f"++inference.tokens_to_generate={args.max_model_len} "
        ),
        cluster=args.cluster,
        expname=args.exp_name,
        model=args.model_path,
        server_gpus=args.num_gpus,
        server_nodes=args.num_nodes,
        server_type="vllm",
        server_args="--async-scheduling",
        benchmarks=args.benchmarks,
        output_dir=args.output,
        data_dir="/workspace/datasets/ns_datasets"
    )