

# Include other necessary imports and any relevant constants

import os
import re
from typing import Tuple
from python.turbine_models.model_builder import HFTransformerBuilder
import safetensors
from shark_turbine.aot import *
from iree.compiler.ir import Context

import torch
from torch.utils import _pytree as pytree
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Literal

from turbine_models.custom_models import remap_gguf





# TODO (Dan): replace this with a file once I figure out paths on windows exe
json_schema = """
[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]
"""



# TODO (Dan): replace this with a file once I figure out paths on windows exe
json_schema = """
[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]
"""


BATCH_SIZE = 1
MAX_STEP_SEQ = 4095

import argparse

parser = argparse.ArgumentParser()
def dir_path(path): # check arguments during parse
    if os.path.isdir(path):
        return path
    else:
        try:
            os.makedirs(path, exist_ok=True)
            return path
        except:
            raise argparse.ArgumentTypeError(f"save_path:{path} is not a valid path")
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token for downloading the module. Optional if you have done `huggingface-cli login`"
)
parser.add_argument("--compile_to", type=str, help="What MLIR IR to save to. Options: torch, linalg")
parser.add_argument("--save_mlir", type=bool, default=True, help="Save mlir file")
parser.add_argument("--save_vmfb", type=bool, default=True, help="Save vmfb file")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="Specifying external weights saves ir/vmfb without global weights for size and readability, options [None, gguf, safetensors, path to safetensors file]",
)
parser.add_argument("--save_path", type=dir_path, default="./")
parser.add_argument("--quantization", type=str, default="None", help="")
parser.add_argument(
    "--precision", type=str, default="fp16", help="dtype of model [f16, f32]"
)


if __name__ == "__main__":
    """
    By default, this runs a chat demo with the model.
    To export vmfb, run with --compile_to vmfb
    To export torch, run with --compile_to torch
    """
    args = parser.parse_args()
    print(args.compile_to)
# class HFTransformerBuilder:
    # """
    # A model builder that uses Hugging Face's transformers library to build a PyTorch model.

    # Args:
    #     example_input (torch.Tensor): An example input tensor to the model.
    #     hf_id (str): The Hugging Face model ID.
    #     auto_model (AutoModel): The AutoModel class to use for loading the model.
    #     auto_tokenizer (AutoTokenizer): The AutoTokenizer class to use for loading the tokenizer.
    #     auto_config (AutoConfig): The AutoConfig class to use for loading the model configuration.
    # """

    # def __init__(
    #     self,
    #     example_input: torch.Tensor,
    #     hf_id: str = None,
    #     auto_model: AutoModel = AutoModel,
    #     auto_tokenizer: AutoTokenizer = None,
    #     auto_config: AutoConfig = None,
    #     hf_auth_token=None,
    #     quantization=None,
    #     precision=None,
    #     external_weights=None,
    #     compile_to: Literal["torch", "linalg", "vmfb"] = "torch", # only applies to llama
    builder = HFTransformerBuilder(
        example_input=torch.tensor([[1]], dtype=torch.int64),
        hf_id=args.hf_model_name,
        hf_auth_token=args.hf_auth_token,
        external_weights=args.external_weights,
        quantization=args.quantization,
        precision=args.precision,
        compile_to=args.compile_to,
    )
    
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if args.save_mlir:
        compiled_module = builder.get_compiled_module(save_to = f"{safe_name}.mlir")
    if args.save_vmfb:
        compiled_module.save_vmfb(f"{safe_name}.vmfb")