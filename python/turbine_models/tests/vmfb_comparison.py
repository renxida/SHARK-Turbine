import os
import sys
import re

from typing import Tuple

os.environ["TORCH_LOGS"] = "dynamic"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils import _pytree as pytree
from shark_turbine.aot import *
from iree.compiler.ir import Context
from iree import runtime as ireert

from turbine_models.custom_models import remap_gguf
import safetensors

from tqdm import tqdm
from typing import Literal

import turbine_models.custom_models.stateless_llama as llama

BATCH_SIZE = 1
MAX_STEP_SEQ = 4095


BATCH_SIZE = 1
MAX_STEP_SEQ = 4095


def torch_token_generator(
    prompt,
    hf_model_name: str,
    hf_auth_token: str,
    break_on_eos=False,
    precision="f32",
    quantization="unquantized",
):
    if precision == "f16":
        torch_dtype = torch.float16
    elif precision == "f32":
        torch_dtype = torch.float32
    else:
        raise ValueError("Invalid dtype, f16 or f32 supported")

    if (
        quantization is not None
        and quantization.lower() != "none"
        and quantization.lower() != "unquantized"
    ):
        raise NotImplementedError("Quantization not supported for torch")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        use_auth_token=hf_auth_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name, torch_dtype=torch_dtype, use_auth_token=hf_auth_token
    )

    initial_input = tokenizer(prompt, return_tensors="pt")
    input_ids = initial_input.input_ids
    past_key_values = None

    while True:
        model_results = model.forward(input_ids, past_key_values=past_key_values)
        logits = model_results.logits
        next_token_id = torch.argmax(logits[:, -1, :], dim=1)
        past_key_values = model_results.past_key_values

        yield next_token_id
        input_ids = next_token_id.unsqueeze(0)  # Prepare for the next iteration

        if next_token_id.item() == tokenizer.eos_token_id and break_on_eos:
            break



def turbine_token_generator(
    prompt: str,
    hf_model_name: str,
    vmfb_path: str = None,
    external_weight_file: str = None,
    hf_auth_token: str = None,
    break_on_eos: bool = False,
    device: Literal["llvm-cpu", "cuda", "vulcan", "rocm"] = "llvm-cpu",
) -> torch.Tensor:
    """
    A generator function for turbine model inference.

    :param prompt: The input prompt for the model.
    :param hf_model_name: The name of the Hugging Face model.
    :param vmfb_path: Path to the .vmfb model file.
    :param external_weight_file: Path to the external weight file (optional).
    :param hf_auth_token: Hugging Face authorization token (optional).
    :param break_on_eos: Whether to break the loop on end-of-sentence token.
    :return: Yields a tensor representing the generated token.
    """

    # Create the config for the IREE runtime environment
    config = ireert.Config("local-task" if device == "llvm-cpu" else device)

    # Load the external weight file if provided
    if external_weight_file:
        index = ireert.ParameterIndex()
        index.load(external_weight_file)

    # Ensure model name is in a safe format
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)

    # Load the .vmfb model file
    if vmfb_path:
        mod = ireert.VmModule.mmap(config.vm_instance, vmfb_path)
    elif os.path.exists(f"{safe_name}.vmfb"):
        mod = ireert.VmModule.mmap(config.vm_instance, f"{safe_name}.vmfb")
    else:
        raise FileNotFoundError("No vmfb_path provided, required for run_vmfb")

    # Prepare the modules for the IREE runtime context
    vm_modules = [mod, ireert.create_hal_module(config.vm_instance, config.device)]

    # Include parameter module if external weight file is used
    if external_weight_file:
        param_module = ireert.create_io_parameters_module(
            config.vm_instance, index.create_provider(scope="model")
        )
        vm_modules.insert(0, param_module)

    # Create the system context with the given configuration and modules
    ctx = ireert.SystemContext(vm_modules=vm_modules, config=config)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name, use_fast=False, use_auth_token=hf_auth_token
    )

    # Convert the prompt to input tensor
    initial_input = tokenizer(prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids
    device_inputs = [ireert.asdevicearray(config.device, example_input_id)]

    # Get the compiled module
    ModuleCompiled = ctx.modules.state_update
    results = ModuleCompiled["run_initialize"](*device_inputs)

    def format_out(results):
        # Convert the output to a PyTorch tensor
        return torch.tensor(results.to_host()[0][0])

    # Token generation loop
    while True:
        next_token_tensor = format_out(results)
        yield next_token_tensor.item()  # Yield the scalar value of the tensor

        # Run the next step of the model
        results = ModuleCompiled["run_forward"](results)

        # Check for the end-of-sentence token
        if next_token_tensor.item() == tokenizer.eos_token_id and break_on_eos:
            break


def get_torch_string(
    prompt,
    hf_auth_token,
    hf_model_name,
    precision,
    quantization,
    tokens_to_compare=50,
):
    print("Using prompt:")
    print(prompt)
    print("To generate torch reference string...")
    torch_gen = torch_token_generator(
        prompt=prompt,
        hf_auth_token=hf_auth_token,
        hf_model_name=hf_model_name,
        break_on_eos=True,
        precision=precision,
        quantization=quantization,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name, use_fast=False, use_auth_token=hf_auth_token
    )

    print(
        "Generating Torch tokens... The pipeline needs to be initialized first so the first few tokens may take a while."
    )
    # read until stopiteration
    torch_tokens = list(tqdm(torch_gen, desc="Generating Torch tokens"))
    torch_str = tokenizer.decode(torch.tensor(torch_tokens).numpy())

    return torch_str


def get_turbine_vmfb_string(
    prompt,
    hf_auth_token,
    hf_model_name,
    vmfb_path,
    external_weight_file,
    device,
    tokens_to_compare=50,
):
    # Initialize generators with the prompt and specific arguments
    # check if torch string cache exists
    # cache is at python/turbine_models/tests/vmfb_comparison_cached_torch_output.txt

    # Decode and print the outputs
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name, use_fast=False, use_auth_token=hf_auth_token
    )

    # Run turbine until an equal number of tokens has been generated
    print(
        "Generating Turbine tokens... The pipeline needs to be initialized first so the first few tokens may take a while."
    )
    turbine_gen = turbine_token_generator(
        prompt=prompt,
        hf_model_name=hf_model_name,
        vmfb_path=vmfb_path,
        external_weight_file=external_weight_file,
        hf_auth_token=hf_auth_token,
        break_on_eos=False,
        device=device,
    )
    turbine_tokens = []
    for _ in tqdm(range(tokens_to_compare), desc="Generating Turbine tokens"):
        token = next(turbine_gen)
        turbine_tokens.append(token)
    del turbine_gen

    turbine_str = tokenizer.decode(torch.tensor(turbine_tokens).numpy())
    return turbine_str



def run_vmfb_comparison(quantization="unquantized", precision="f32", device="llvm-cpu"):
    """
    Test that the vmfb model produces the same output as the torch model

    Precision can be 16 or 32, using 16 for speed and memory.

    For VMFB, quantization can be int4 or None, but right now only using none for compatibility with torch.
    """
    quantization = "unquantized"
    precision = "f32"

    llama.export_transformer_model(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        hf_auth_token=None,
        compile_to="vmfb",
        external_weights="safetensors",
        # external_weight_file="Llama-2-7b-chat-hf-function-calling-v2_f16_int4.safetensors", Do not export weights because this doesn't get quantized
        quantization=quantization,
        precision=precision,
        device=device,
        target_triple="host",
    )

    from turbine_models.gen_external_params.gen_external_params import (
        gen_external_params,
    )

    gen_external_params(
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        quantization=quantization,
        hf_auth_token=None,
        precision=precision,   
    )

    DEFAULT_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""

    torch_str_cache_path = f"python/turbine_models/tests/vmfb_comparison_cached_torch_output_{precision}_{quantization}.txt"
    # if cached, just read
    if os.path.exists(torch_str_cache_path):
        with open(torch_str_cache_path, "r") as f:
            torch_str = f.read()
    else:
        from .vmfb_comparison import get_torch_string

        torch_str = get_torch_string(
            prompt=DEFAULT_PROMPT,
            hf_auth_token=None,
            hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
            tokens_to_compare=50,
            precision=precision,
            quantization=quantization,
        )

        with open(torch_str_cache_path, "w") as f:
            f.write(torch_str)

    turbine_str = get_turbine_vmfb_string(
        prompt=DEFAULT_PROMPT,
        hf_auth_token=None,
        hf_model_name="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        vmfb_path="Llama_2_7b_chat_hf_function_calling_v2.vmfb",
        external_weight_file=f"Llama_2_7b_chat_hf_function_calling_v2_{precision}_{quantization}.safetensors",
        tokens_to_compare=50,
        device=device,
    )

    torch_str = torch_str[: len(turbine_str)]

    import difflib

    # Calculate and print diff
    diff = difflib.unified_diff(
        torch_str.splitlines(keepends=True),
        turbine_str.splitlines(keepends=True),
        fromfile="torch_str",
        tofile="turbine_str",
        lineterm="",
    )

    assert torch_str == turbine_str, "".join(diff)