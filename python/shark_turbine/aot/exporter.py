# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Xida: todo: remove redundant imports
import os
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
from .builtins import *

from transformers.models.llama.modeling_llama import LlamaForCausalLM
# end Xida imports

from typing import Literal, Optional, Sequence, Union
import functools
import io
from pathlib import Path
import platform

import torch

from iree.compiler.api import (
    Invocation,
    Session,
    Source,
    Output,
)

from .compiled_module import (
    CompiledModule,
    CompiledModuleMeta,
    ExportProcDef,
)
from .support.ir_imports import (
    Context,
    Operation,
)
from .support.procedural import (
    AbstractTypedef,
)


_is_windows = platform.system() == "Windows"


ModuleLike = Union[torch.nn.Module, CompiledModuleMeta]
SaveableTarget = Union[str, Path, None, Output]


class ExportOutput:
    """Wrapper around a CompiledModule produced by `export`."""

    def __init__(
        self,
        session: Session,
        compiled_module: CompiledModule,
        *,
        importer_uses_session: bool = False,
    ):
        self.session = session
        self.session.set_flags("--iree-input-type=torch")
        self.compiled_module = compiled_module
        self._importer_uses_session = importer_uses_session

    @property
    def mlir_module(self) -> Operation:
        """Gets the MLIR module resulting from the last compilation phase."""
        return CompiledModule.get_mlir_module(self.compiled_module)

    def print_readable(self, large_elements_limit: int = 50):
        """Prints a human readable version of the current compilation IR."""
        self.mlir_module.print(large_elements_limit=large_elements_limit)

    def save_mlir(self, file_path: Union[str, Path]):
        """Saves the current compilation IR to a path on disk.

        Args:
            file_path: Path to save the file. If it has a ".mlirbc"
              extension, it will be saved as bytecode. Otherwise as
              text.
        """
        file_path = Path(file_path)
        with open(file_path, "wb") as f:
            if file_path.suffix == ".mlirbc":
                self.mlir_module.write_bytecode(f)
            else:
                self.mlir_module.print(f, binary=True)

    def _run_import(self):
        CompiledModule.run_import(self.compiled_module)

    def compile(
        self,
        save_to: SaveableTarget,
        *,
        target_backends: Union[str, Sequence[str]] = ("llvm-cpu",),
    ) -> Optional[memoryview]:
        """Compiles the exported program to an executable binary.

        Args:
            save_to: Where to save the compiled binary. Can be one of:
              None: outputs to a memory buffer and return the API Output.
              (str, Path): Outputs to a file
              Output: Raw compiler API Output object to save to.
            target_backends: A comma-delimitted string of IREE target backends or
              a sequence of strings.
        Returns:
          None unless if `save_to=None`, in which case, we return the backing compiler API
          Ouptut object. It can be queried for its backing memory via its `map_memory()`
          method.
        """
        return_memory_view = False
        if save_to is None:
            output = Output.open_membuffer()
            return_memory_view = True
        elif isinstance(save_to, (str, Path)):
            save_to = Path(save_to)
            output = Output.open_file(str(save_to))
        else:
            assert isinstance(output, Output)
            output = save_to

        target_backends = (
            target_backends
            if isinstance(target_backends, str)
            else ",".join(target_backends)
        )
        inv = self.session.invocation()
        if self._importer_uses_session:
            inv.import_module(self.mlir_module)
        else:
            # Some platforms can't share the context across the importer and
            # session (cough: Windows). Round-trip in this case.
            buffer_io = io.BytesIO()
            self.mlir_module.write_bytecode(buffer_io)
            buffer = buffer_io.getvalue()
            source = Source.wrap_buffer(self.session, buffer)
            inv.parse_source(source)
        inv.enable_console_diagnostics()

        # TODO: Don't use flags to set the target backends: set module attributes.
        self.session.set_flags(f"--iree-hal-target-backends={target_backends}")
        if not inv.execute():
            raise RuntimeError("Compilation failed: See diagnostics")

        inv.output_vm_bytecode(output)
        output.keep()
        if return_memory_view:
            return output
        else:
            return None


# Decorator which explicitly exports a function.
# TODO: Make this a public API on CompiledModule.
# See https://github.com/nod-ai/SHARK-Turbine/issues/126
def export_proc(f=None, *, signature: Sequence[AbstractTypedef]) -> ExportProcDef:
    if f is None:
        return functools.partial(export_proc, signature=signature)
    return ExportProcDef(f.__name__, f, signature=signature)


def export(mdl: ModuleLike, *example_args: torch.Tensor) -> ExportOutput:
    """One shot export of an nn.Module.

    This is a very restrictive API vs the lower level `CompiledModule`
    facility. It is suitable for one-shot modules, with a single
    entrypoint and static example arguments where no additional
    configuration is needed for mutable parameters/buffers or state
    management. Dynamic shape constraints are also not presently
    exposed via this API, but we expect to allow this in the future.

    Args:
      mdl: The nn.Module to export.
      *example_args: Example tensors.

    Returns:
      An ExportOutput object that wraps the compilation and provides
      easy access.
    """
    if isinstance(mdl, torch.nn.Module):
        signature = [abstractify(t) for t in example_args]

        class Exported(CompiledModule, export_name=mdl._get_name()):
            params = export_parameters(mdl)

            @export_proc(signature=signature)
            def main(self, *args):
                return jittable(mdl.forward)(*args)

    else:
        assert isinstance(mdl, CompiledModuleMeta)
        Exported = mdl

    session = Session()
    # There are some bugs with respect to Session/context interop that we
    # haven't squashed yet. For now, default everyone to round-tripping
    # via bytecode vs sharing the context between the importer/compiler.
    importer_uses_session = False and not _is_windows
    if importer_uses_session:
        context = session.context
    else:
        context = Context()

    cm = Exported(context=context, import_to="import")
    return ExportOutput(session, cm, importer_uses_session=importer_uses_session)



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

def export_llama(
    mod: LlamaForCausalLM,
    compile_to: Literal["torch", "linalg", "vmfb"] = "torch",
    hf_auth_token=None,
    external_weights=None,
    quantization=None,
    precision=None,
    *example_args: torch.Tensor,
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
    json_schema = """
    [1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]
    """
    state_schema = pytree.treespec_loads(json_schema)

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
