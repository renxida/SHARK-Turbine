

# Include other necessary imports and any relevant constants

import re
from typing import Tuple
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


BATCH_SIZE = 1
MAX_STEP_SEQ = 4095



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


def save_vmfb(module_str, vmfb_file):
    flags = [
    "--iree-input-type=torch",
    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
    "--mlir-print-debuginfo",
    "--mlir-print-op-on-diagnostic=false",
    "--iree-llvmcpu-target-cpu-features=host",
    "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
    "--iree-llvmcpu-enable-microkernels",
    "--iree-llvmcpu-stack-allocation-limit=256000",
    "--iree-stream-resource-index-bits=64",
    "--iree-vm-target-index-bits=64",
    "--iree-vm-bytecode-module-strip-source-map=true",
    "--iree-util-zero-fill-elided-attrs",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-codegen-check-ir-before-llvm-conversion=false",
    "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
    "--iree-opt-const-expr-hoisting=False",
    ]

    import iree.compiler as ireec

    flatbuffer_blob = ireec.compile_str(
        module_str,
        target_backends=["llvm-cpu"],
        extra_args=flags,
    )
    with open(vmfb_file, "wb+") as f:
        f.write(flatbuffer_blob)
    print("saved to ", vmfb_file + ".vmfb")


def compile_hf_transformer_model(
    hf_model_name,
    compile_to: Literal["torch", "linalg", "vmfb"] = "torch",
    hf_auth_token=None,
    external_weights=None,
    external_weight_file=None,
    quantization=None,
    precision=None,
)  -> Tuple[str, AutoTokenizer]:
    """
    Exports the transformer model specified by hf_model_name to a chosen format.

    Args:
        hf_model_name (str): Name of the Hugging Face model to be exported.
        hf_auth_token (str, optional): Authentication token for Hugging Face.
        compile_to (str, optional): Target compilation format. Can be one of the following: Literal["torch", "linalg", "vmfb"]
        external_weights (str, optional): Specifies the handling of external weights.
        external_weight_file (str, optional): Path to the external weight file.
        quantization (str, optional): Type of quantization to apply.
        precision (str, optional): Precision of the model parameters (e.g., 'fp16').

    Returns:
        Tuple[str, AutoTokenizer]: A tuple containing the model compiled to the specified MLIR string, and the associated tokenizer.
        None: if compile_to == "vmfb". Will save to vmfb file.
    """
    print("Compiling model to", compile_to)
    state_schema = pytree.treespec_loads(json_schema)

    mod = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float,
        token=hf_auth_token,
    )
    dtype = torch.float32
    if precision == "f16":
        mod = mod.half()
        dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        token=hf_auth_token,
    )
    # TODO: generate these values instead of magic numbers
    HEADS = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 1
    global_pkv = torch.zeros(
        size=(HEADS * 2, BATCH_SIZE, MAX_STEP_SEQ, HEADS, HIDDEN_DIM),
        dtype=dtype,
    )

    mapper = {}
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(mod.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            if external_weight_file:
                safetensors.torch.save_file(mod_params, external_weight_file)

        elif external_weights == "gguf":
            tensor_mapper = remap_gguf.TensorNameMap(remap_gguf.MODEL_ARCH.LLAMA, HEADS)
            mapper = tensor_mapper.mapping

    class StateUpdateModule(CompiledModule):
        """
        The class handles operations related to updating and slicing the global state of the model.
        Inherits from shark_turbine.aot.compiled_module.CompiledModule
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

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    pre_import_passes = []
    if quantization == "int4" and not compile_to == "linalg":
        from shark_turbine.transforms.quantization import mm_group_quant
        pre_import_passes.append(mm_group_quant.MMGroupQuantRewriterPass)
    inst = StateUpdateModule(context=Context(), import_to=import_to, pre_import_passes=pre_import_passes)
    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)


    if compile_to == "vmfb":
        save_vmfb(module_str, safe_name+".vmfb")
        return module_str, tokenizer
        
    return module_str, tokenizer



