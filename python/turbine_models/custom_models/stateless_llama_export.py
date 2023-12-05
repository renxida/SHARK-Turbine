"""
This is a modified version of the export function from shark_turbine.aot.exporter.py.

The main difference is that it uses the StateUpdateModule class
instead of the CompiledModule class.

TODO: make this handle different language models and merge it into shark_turbine.aot.exporter
"""

import os
import re
from typing import Tuple, Literal, Optional, Sequence, Union
import safetensors
from shark_turbine.aot import *
from iree.compiler.ir import Context

import torch
from torch.utils import _pytree as pytree
from transformers import AutoModelForCausalLM, AutoTokenizer

from turbine_models.custom_models import remap_gguf
from shark_turbine.aot.builtins import *

from transformers.models.llama.modeling_llama import LlamaForCausalLM

import functools
import io
from pathlib import Path
import platform

from iree.compiler.api import (
    Invocation,
    Session,
    Source,
    Output,
)

from shark_turbine.aot.compiled_module import (
    CompiledModule,
    CompiledModuleMeta,
    ExportProcDef,
)
from shark_turbine.aot.support.ir_imports import (
    Context,
    Operation,
)
from shark_turbine.aot.support.procedural import (
    AbstractTypedef,
)
from shark_turbine.aot.exporter import ExportOutput, export_proc
from turbine_models.model_builder import HFTransformerBuilder
_is_windows = platform.system() == "Windows"


def slice_up_to_step(global_pkv: torch.Tensor, seq_step: int, heads: int, hidden_dim: int):
    """
    Slices the provided tensor up to the specified sequence step for each head.

    Args:
        global_pkv (Tensor): The global past key values tensor.
        seq_step (int): The current sequence step to slice up to.
        heads (int): The number of attention heads in the model.
        hidden_dim (int): The dimension of the hidden layer.

    Returns:
        List[Tensor]: A list of sliced tensors for each head.
    """
    all_pkv_tensors = []
    for i in range(heads * 2):
        sliced = IREE.tensor_slice(
            global_pkv, i, 0, (0, seq_step), (0, heads), (0, hidden_dim)
        )  # sequence context dim
        all_pkv_tensors.append(
            IREE.tensor_reshape(sliced, 1, seq_step, heads, hidden_dim)
        )

    return all_pkv_tensors

ModuleLike = Union[torch.nn.Module, CompiledModuleMeta]
SaveableTarget = Union[str, Path, None, Output]
def stateless_llama_export(
    mod: LlamaForCausalLM,
    example_args: torch.Tensor,
    compile_to: Literal["torch", "linalg", "vmfb"] = "torch",
    hf_auth_token=None,
    external_weights=None,
    quantization=None,
    precision=None,
) -> ExportOutput:
    """
    Exports the transformer model specified by hf_model_name to a chosen format.

    Args:
        hf_model_name (str): Name of the Hugging Face model to be exported.
        compile_to (str, optional): Target compilation format. Can be one of the following: Literal["torch", "linalg", "vmfb"]
        external_weights (str, optional): Specifies the handling of external weights.
        external_weight_file (str, optional): Path to the external weight file.
        quantization (str, optional): Type of quantization to apply.
        precision (str, optional): Precision of the model parameters (e.g., 'fp16').

    Returns:
        Tuple[str, AutoTokenizer]: A tuple containing the model compiled to the specified MLIR string, and the associated tokenizer.
        None: if compile_to == "vmfb". Will save to vmfb file.
    """
    signature = [abstractify(t) for t in example_args]
    print("Compiling model to", compile_to)
    # TODO (Dan): replace this with a file once I figure out paths on windows exe
    from .stateless_llama_state_schema import state_schema

    dtype = torch.float32
    if precision == "f16":
        mod = mod.half()
        dtype = torch.float16
        
    # TODO: generate these values instead of magic numbers
    MAX_STEP_SEQ = 4095

    HEADS = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 1
    global_pkv = torch.zeros(
        size=(HEADS * 2, BATCH_SIZE, MAX_STEP_SEQ, HEADS, HIDDEN_DIM),
        dtype=dtype,
    )

    mapper = {}
    if external_weights is not None:
        if external_weights == "safetensors" or external_weights.endswith("safetensors"):
            mod_params = dict(mod.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            # check if there's a file
            if external_weights.endswith(".safetensors"):
                external_weights_file = external_weights
                safetensors.torch.save_file(mod_params, external_weights_file)

        elif external_weights == "gguf":
            tensor_mapper = remap_gguf.TensorNameMap(remap_gguf.MODEL_ARCH.LLAMA, HEADS)
            mapper = tensor_mapper.mapping

    class StateUpdateModule(CompiledModule):
        """
        The class handles operations related to updating and slicing the global state of the model.
        Inherits from shark_turbine.aot.compiled_module.CompiledModule.

        It has to be a class-in-a-function because the jittable functions
        `initialize()` and `forward` need to be static but also depends on mod.

        There is probably a way to refactor this to take in `mod` via __new__()
        """
        if external_weights:
            params = export_parameters(
                mod, external=True, external_scope="", name_mapper=mapper.get
            )
        else:
            params = export_parameters(mod)
        global_state = export_global(
            abstractify(global_pkv), uninitialized=True, mutable=True
        )
        global_seq_step = export_global(AbstractIndex, mutable=True)

        def run_initialize(self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)):
            init_const = [x.dynamic_dim(1) < MAX_STEP_SEQ]
            token, *state = self.initialize(x, constraints=init_const)
            self.global_seq_step = IREE.tensor_dim(
                state[0], 1
            )  # ? dimension of arbitrarily 0th kv tensor
            for i in range(HEADS * 2):
                slice_of_state = IREE.tensor_reshape(
                    state[i], 1, 1, self.global_seq_step, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, slice_of_state, i, 0, 0, 0, 0
                )
            return token

        def run_forward(self, x=AbstractTensor(1, 1, dtype=torch.int64)):
            state_arg = slice_up_to_step(
                self.global_state, self.global_seq_step, HEADS, HIDDEN_DIM
            )
            forw_const = (
                [state_arg[0].dynamic_dim(1) < MAX_STEP_SEQ]
                + [
                    x.dynamic_dim(1) == (state_arg[0].dynamic_dim(1))
                    for x in state_arg[1:]
                ]
                + [x.dynamic_dim(1) < MAX_STEP_SEQ for x in state_arg[1:]]
            )
            token, *state_update = self.forward(x, *state_arg, constraints=forw_const)
            for i in range(HEADS * 2):
                update = IREE.tensor_reshape(
                    state_update[i], 1, 1, 1, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, update, i, 0, self.global_seq_step, 0, 0
                )

            self.global_seq_step = self.global_seq_step + 1
            return token

        def get_global_state(self):
            return self.global_state

        def get_seq_step(self):
            return self.global_seq_step

        @jittable
        def initialize(input_ids):
            result = mod.forward(input_ids)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            state1_flat = [torch.transpose(x, 1, 2) for x in state1_flat]
            return token1, *state1_flat

        @jittable
        def forward(token0: torch.Tensor, *state0_flat):
            # Unpad the states.
            state0_flat = [torch.transpose(x, 1, 2) for x in state0_flat]
            state0 = pytree.tree_unflatten(state0_flat, state_schema)
            result = mod.forward(token0, past_key_values=state0)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            state1_flat = [torch.transpose(x[:, :, -1:, :], 1, 2) for x in state1_flat]
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            return token1, *state1_flat
        

        @export_proc(signature=signature)
        def main(self, *args):
            return jittable(mod.forward)(*args)


    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    pre_import_passes = []
    if quantization == "int4":
        from shark_turbine.transforms.quantization import mm_group_quant
        pre_import_passes.append(mm_group_quant.MMGroupQuantRewriterPass)
    compiled_module = StateUpdateModule(context=Context(), import_to=import_to, pre_import_passes=pre_import_passes)

    session = Session()
    # There are some bugs with respect to Session/context interop that we
    # haven't squashed yet. For now, default everyone to round-tripping
    # via bytecode vs sharing the context between the importer/compiler.
    importer_uses_session = False and not _is_windows
    if importer_uses_session:
        context = session.context
    else:
        context = Context()

    return ExportOutput(session, compiled_module, importer_uses_session=importer_uses_session)




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
    default="llSourcell/medllama2_7b",
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
parser.add_argument("--quantization", type=str, default="None", help="Quantization. Options: [None, int4]")
parser.add_argument(
    "--precision", type=str, default="fp16", help="dtype of model [f16, f32]"
)
parser.add_argument("--target_backends", type=list, default=["llvm-cpu"], help="VMFB supported target backends.")


if __name__ == "__main__":
    """
    By default, this runs a chat demo with the model.
    To export vmfb, run with --compile_to vmfb
    To export torch, run with --compile_to torch
    """
    args = parser.parse_args()
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
    elif args.save_vmfb:
        compiled_module = builder.get_compiled_module()
    if args.save_vmfb:
        compiled_module.save_vmfb(f"{safe_name}.vmfb", target_backends=args.target_backends)