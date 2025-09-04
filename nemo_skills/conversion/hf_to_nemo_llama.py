# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

# copied from https://github.com/NVIDIA/NeMo/blob/main/scripts/nlp_language_modeling/convert_hf_llama_to_nemo.py


import os
import shutil
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging
from omegaconf import OmegaConf
from pytorch_lightning.trainer.trainer import Trainer
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--in-path",
        type=str,
        default=None,
        required=True,
        help="Path to Huggingface LLaMA checkpoints",
    )
    parser.add_argument(
        "--out-path", type=str, default=None, required=True, help="Path to output nemo folder (untarred)."
    )
    parser.add_argument("--precision", type=str, default="16", help="Model precision")
    parser.add_argument(
        # this is required for Llama3 tokenizer loading as it's not in the checkpoint dir
        "--hf-model-name",
        required=False,
        help="Name of HF model we are converting to (e.g. mistralai/Mistral-7B-v0.1)",
    )
    parser.add_argument("--override", action="store_true", help="Override existing output directory if it exists.")
    parser.add_argument("--nemo-format", choices=["zarr", "torch_dist"], default="zarr", help="NeMo checkpoint format")
    args = parser.parse_args()

    return args


def load_config(llama_config):
    nemo_config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "nemo_config_llama.yaml")).model
    if llama_config.get("rope_theta", None):
        nemo_config["rotary_base"] = llama_config["rope_theta"]
    nemo_config.encoder_seq_length = llama_config["max_position_embeddings"]
    nemo_config.num_layers = int(llama_config["num_hidden_layers"])
    nemo_config.hidden_size = llama_config["hidden_size"]
    nemo_config.ffn_hidden_size = llama_config["intermediate_size"]
    nemo_config.num_attention_heads = llama_config["num_attention_heads"]
    nemo_config.max_position_embeddings = llama_config["max_position_embeddings"]
    nemo_config.init_method_std = llama_config["initializer_range"]
    nemo_config.layernorm_epsilon = llama_config["rms_norm_eps"]

    # for mistral model
    if "sliding_window" in llama_config:
        nemo_config.window_size = [llama_config["sliding_window"], 0]

    if "num_key_value_heads" in llama_config:
        nemo_config.num_query_groups = llama_config["num_key_value_heads"]
    nemo_config.use_cpu_initialization = True
    nemo_config.activation = "fast-swiglu"

    # Tokenizer config
    if "tokenizer_model" in llama_config:
        nemo_config.tokenizer.model = llama_config["tokenizer_model"]
    else:
        # Llama3 uses converted TikToken Tokenizer
        tokenizer_dict = {
            "library": "huggingface",
            "type": args.hf_model_name,
            "use_fast": True,
        }
        nemo_config.tokenizer = tokenizer_dict

    if llama_config["rope_scaling"] is not None:
        rope_type = llama_config["rope_scaling"].get("rope_type")
        if rope_type is None:
            rope_type = llama_config["rope_scaling"].get("type")
        if rope_type in ("linear", "llama3"):
            # TODO: some bug with that parameter?
            # nemo_config['seq_len_interpolation_factor'] = llama_config['rope_scaling']['factor']
            if rope_type == "llama3":
                nemo_config.scale_positional_embedding = True
        else:
            raise ValueError("Only linear rope scaling type is supported now")

    base = 128
    while llama_config["vocab_size"] % base != 0:
        base //= 2
    nemo_config.make_vocab_size_divisible_by = base
    nemo_config.dist_ckpt_format = args.nemo_format
    return nemo_config


def load_state_dict_helper(cls, cfg, trainer: Trainer, state_dict):
    """Load state_dict for converted community, for example, HuggingFace models."""
    model = cls(cfg, trainer)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        # Keys ending with '_extra_state' are related to Transformer Engine internals
        missing_keys_non_extra = [key for key in missing_keys if not key.endswith("_extra_state")]
        if missing_keys_non_extra:
            logging.critical("Missing keys were detected during the load, something has gone wrong. Aborting.")
            raise RuntimeError(f"Missing keys: \n{missing_keys_non_extra}")

    if unexpected_keys:
        logging.critical("Unexpected keys were detected which should not happen. Aborting.")
        raise RuntimeError(f"Unexpected keys: \n{unexpected_keys}")

    return model


def convert(args):
    logging.info(f"loading checkpoint {args.in_path}")

    model = LlamaForCausalLM.from_pretrained(args.in_path)
    hf_config = vars(model.config)
    if os.path.exists(f"{args.in_path}/tokenizer.model"):
        tokenizer = LlamaTokenizer.from_pretrained(args.in_path)
        hf_config["tokenizer_model"] = str(tokenizer.vocab_file)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_name)

    print(f"hf_config: {hf_config}")
    print("named parameters:")
    for name, param in model.named_parameters():
        print(f"- {name}")

    nemo_config = load_config(hf_config)

    if args.precision in ["32", "16"]:
        precision = int(float(args.precision))
    elif args.precision in ["bf16", "bf16-mixed"]:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = args.precision
        else:
            logging.warning("BF16 is not supported on this device. Using FP16 instead.")
            precision = args.precision[2:]  # prune bf in string
    else:
        precision = args.precision

    plugins = []
    if precision in [16, "16", "bf16", "16-mixed", "bf16-mixed"]:
        scaler = None
        if precision in [16, "16", "16-mixed"]:
            scaler = GradScaler(
                init_scale=nemo_config.get("native_amp_init_scale", 2**32),
                growth_interval=nemo_config.get("native_amp_growth_interval", 1000),
                hysteresis=nemo_config.get("hysteresis", 2),
            )
            # MixedPrecisionPlugin in PTL >= 2.0 requires precision to be 16-mixed or bf16-mixed
            plugin_precision = "16-mixed"
        else:
            plugin_precision = "bf16-mixed"

        if nemo_config.get("megatron_amp_O2", False):
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device="cuda", scaler=scaler))

    nemo_config.precision = precision
    print(f"nemo_config: {nemo_config}")

    trainer = Trainer(plugins=plugins, accelerator="cpu", strategy=NLPDDPStrategy())

    hidden_size = hf_config["hidden_size"]
    head_num = hf_config["num_attention_heads"]
    head_size = hidden_size // head_num
    num_layers = hf_config["num_hidden_layers"]

    mcore_gpt = nemo_config.mcore_gpt

    assert mcore_gpt == nemo_config.get("transformer_engine", False), (
        "mcore_gpt transformer_engine must be enabled (or disabled) together."
    )

    def param_to_weights(param):
        return param.float()

    checkpoint = OrderedDict()
    checkpoint["state_dict"] = OrderedDict()

    embed_weight = model.state_dict()["model.embed_tokens.weight"]
    if mcore_gpt:
        embed_weights_base_name = "model.embedding.word_embeddings.weight"
    else:
        embed_weights_base_name = "model.language_model.embedding.word_embeddings.weight"
    checkpoint["state_dict"][embed_weights_base_name] = param_to_weights(embed_weight)

    # in hf, this is defined as register_buffer(..., persistent=False) so it won't be in the state dict
    if "model.layers.0.self_attn.rotary_emb.inv_freq" in model.state_dict():
        rotary_embed_weight = model.state_dict()["model.layers.0.self_attn.rotary_emb.inv_freq"]
        if mcore_gpt:
            rotary_embed_weight_base_name = "model.rotary_pos_emb.inv_freq"
        else:
            rotary_embed_weight_base_name = "model.language_model.rotary_pos_emb.inv_freq"
        checkpoint["state_dict"][rotary_embed_weight_base_name] = param_to_weights(rotary_embed_weight)

    if nemo_config.num_query_groups is None or nemo_config.num_query_groups == head_num:
        num_query_groups = head_num
    else:
        num_query_groups = nemo_config.num_query_groups
        assert head_num % num_query_groups == 0, "head_num must be divisible by num_query_groups"
    if mcore_gpt:
        assert nemo_config.activation.startswith("fast-"), "mcore only supports fast version of gated linear unit."

    for layer_idx in range(int(num_layers)):
        print(f"converting layer {layer_idx}")
        old_tensor_shape = model.state_dict()[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].size()
        new_q_tensor_shape = (head_num, head_size) + old_tensor_shape[1:]
        new_kv_tensor_shape = (num_query_groups, head_size) + old_tensor_shape[1:]
        q = model.state_dict()[f"model.layers.{layer_idx}.self_attn.q_proj.weight"].view(*new_q_tensor_shape)
        k = model.state_dict()[f"model.layers.{layer_idx}.self_attn.k_proj.weight"].view(*new_kv_tensor_shape)
        v = model.state_dict()[f"model.layers.{layer_idx}.self_attn.v_proj.weight"].view(*new_kv_tensor_shape)
        qkv_weights = torch.empty((0, head_size) + old_tensor_shape[1:])
        heads_per_group = head_num // num_query_groups
        for i in range(num_query_groups):
            qkv_weights = torch.cat((qkv_weights, q[i * heads_per_group : (i + 1) * heads_per_group, :, :]))
            qkv_weights = torch.cat((qkv_weights, k[i : i + 1, :, :]))
            qkv_weights = torch.cat((qkv_weights, v[i : i + 1, :, :]))
        qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
        if mcore_gpt:
            qkv_weights_base_name = f"model.decoder.layers.{layer_idx}.self_attention.linear_qkv.weight"
        else:
            qkv_weights_base_name = (
                f"model.language_model.encoder.layers.{layer_idx}.self_attention.query_key_value.weight"
            )
        checkpoint["state_dict"][qkv_weights_base_name] = param_to_weights(qkv_weights)

        # attention dense
        o_weight = model.state_dict()[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
        if mcore_gpt:
            o_weight_base_name = f"model.decoder.layers.{layer_idx}.self_attention.linear_proj.weight"
        else:
            o_weight_base_name = f"model.language_model.encoder.layers.{layer_idx}.self_attention.dense.weight"
        checkpoint["state_dict"][o_weight_base_name] = param_to_weights(o_weight)

        # MLP
        mlp_down_weight = model.state_dict()[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        mlp_gate_weight = model.state_dict()[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        if mcore_gpt:
            mlp_down_base_name = f"model.decoder.layers.{layer_idx}.mlp.linear_fc1.weight"
        else:
            mlp_down_base_name = f"model.language_model.encoder.layers.{layer_idx}.mlp.dense_h_to_4h.weight"
        mlp_down_weight = torch.cat((mlp_down_weight, mlp_gate_weight), axis=0)
        checkpoint["state_dict"][mlp_down_base_name] = param_to_weights(mlp_down_weight)

        mlp_up_weight = model.state_dict()[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
        if mcore_gpt:
            mlp_up_base_name = f"model.decoder.layers.{layer_idx}.mlp.linear_fc2.weight"
        else:
            mlp_up_base_name = f"model.language_model.encoder.layers.{layer_idx}.mlp.dense_4h_to_h.weight"
        checkpoint["state_dict"][mlp_up_base_name] = param_to_weights(mlp_up_weight)

        # LayerNorm
        input_ln_weight = model.state_dict()[f"model.layers.{layer_idx}.input_layernorm.weight"]
        if mcore_gpt:
            input_ln_base_name = f"model.decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight"
        else:
            input_ln_base_name = f"model.language_model.encoder.layers.{layer_idx}.input_layernorm.weight"
        checkpoint["state_dict"][input_ln_base_name] = param_to_weights(input_ln_weight)

        post_attn_ln_weight = model.state_dict()[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
        if mcore_gpt:
            post_attn_ln_base_name = f"model.decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"
        else:
            post_attn_ln_base_name = f"model.language_model.encoder.layers.{layer_idx}.post_attention_layernorm.weight"
        checkpoint["state_dict"][post_attn_ln_base_name] = param_to_weights(post_attn_ln_weight)

        print(f"done layer {layer_idx}")

    final_ln_weight = model.state_dict()["model.norm.weight"]
    if mcore_gpt:
        final_ln_base_name = "model.decoder.final_layernorm.weight"
    else:
        final_ln_base_name = "model.language_model.encoder.final_layernorm.weight"
    checkpoint["state_dict"][final_ln_base_name] = param_to_weights(final_ln_weight)

    output_layer_weight = model.state_dict()["lm_head.weight"]
    if mcore_gpt:
        output_layer_base_name = "model.output_layer.weight"
    else:
        output_layer_base_name = "model.language_model.output_layer.weight"
    checkpoint["state_dict"][output_layer_base_name] = param_to_weights(output_layer_weight)

    checkpoint[MegatronGPTModel.CHECKPOINT_HYPER_PARAMS_KEY] = nemo_config

    del model

    if nemo_config.get("megatron_amp_O2", False):
        keys = list(checkpoint["state_dict"].keys())
        for key in keys:
            checkpoint["state_dict"][key.replace("model.", "model.module.", 1)] = checkpoint["state_dict"].pop(key)

    model = load_state_dict_helper(MegatronGPTModel, nemo_config, trainer, checkpoint["state_dict"])

    model._save_restore_connector = NLPSaveRestoreConnector()

    # cast to target precision and disable cpu init
    dtype = torch_dtype_from_precision(precision)
    model = model.to(dtype=dtype)
    model.cfg.use_cpu_initialization = False
    model._save_restore_connector.pack_nemo_file = False
    # removing out_path if it exists to avoid error
    if args.override:
        try:
            shutil.rmtree(args.out_path)
        except FileNotFoundError:
            pass
    # Adding a dummy model filename here conforms with SaveRestoreConnector's convention
    model.save_to(os.path.join(args.out_path, "model.nemo"))
    logging.info(f"NeMo model saved to: {args.out_path}")


if __name__ == "__main__":
    args = get_args()
    convert(args)
