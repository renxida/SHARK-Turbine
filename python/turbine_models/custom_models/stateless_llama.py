import os
import sys
import re

from typing import Tuple

from turbine_models.model_builder import HFTransformerBuilder

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

BATCH_SIZE = 1

class StatelessLLAMALanguageModel:
    def __init__(self, hf_model_name: str, vmfb_path: str = None, external_weight_file: str = None, hf_auth_token: str = None):
        """
        Initializes the chatbot with the given model and paths.

        :param hf_model_name: The name of the Hugging Face model.
        :param vmfb_path: Path to the .vmfb model file.
        :param external_weight_file: Path to the external weight file (optional).
        :param hf_auth_token: Hugging Face authorization token (optional).

        Xida: maybe make this match the transformers model interface?
        """
        self.hf_model_name = hf_model_name
        self.vmfb_path = vmfb_path
        self.external_weight_file = external_weight_file
        self.hf_auth_token = hf_auth_token

        # Initialize the IREE runtime environment
        self.config = ireert.Config("local-task")
        
        # Load the external weight file if provided
        if self.external_weight_file:
            self.index = ireert.ParameterIndex()
            self.index.load(self.external_weight_file)

        # Load the .vmfb model file
        safe_name = hf_model_name.split("/")[-1].strip()
        safe_name = re.sub("-", "_", safe_name)
        if vmfb_path:
            self.mod = ireert.VmModule.mmap(self.config.vm_instance, vmfb_path)
        elif os.path.exists(f"{safe_name}.vmfb"):
            self.mod = ireert.VmModule.mmap(self.config.vm_instance, f"{safe_name}.vmfb")
        else:
            raise FileNotFoundError("No vmfb_path provided, required for run_vmfb")

        # Prepare the modules for the IREE runtime context
        self.vm_modules = [
            self.mod,
            ireert.create_hal_module(self.config.vm_instance, self.config.device)
        ]

        # Include parameter module if external weight file is used
        if self.external_weight_file:
            param_module = ireert.create_io_parameters_module(
                self.config.vm_instance, self.index.create_provider(scope="model")
            )
            self.vm_modules.insert(0, param_module)

        # Create the system context with the given configuration and modules
        self.ctx = ireert.SystemContext(vm_modules=self.vm_modules, config=self.config)

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=False, use_auth_token=self.hf_auth_token)


    def initialize_generation(self, prompt: str):
        """
        Initializes the token generation process.

        :param prompt: The input prompt for the model.
        """
        initial_input = self.tokenizer(prompt, return_tensors="pt")
        example_input_id = initial_input.input_ids
        device_inputs = [ireert.asdevicearray(self.config.device, example_input_id)]

        self.compiled_module = self.ctx.modules.state_update
        self.next_token = self.compiled_module["run_initialize"](*device_inputs)

    def put(self, token: torch.Tensor):
        """
        Processes a single token.

        :param token: The token to process.
        """
        self.next_token = self.compiled_module["run_forward"](token)

    def get(self) -> torch.Tensor:
        """
        Generates the next token.

        :return: The next token as a torch tensor.
        """
        # get current next token
        next_token_tensor = torch.tensor(self.next_token.to_host()[0][0].item())
        # step
        self.next_token = self.compiled_module["run_forward"](self.next_token)
        return next_token_tensor
    
    def chat(self, prompt, max_tokens_per_message = 30):
        history = []
        self.initialize_generation(prompt)
        for _ in range(max_tokens_per_message):
            token = self.get()
            if token == self.tokenizer.eos_token_id:
                break
        result_output = self.tokenizer.decode(history)
        yield result_output


if __name__ == "__main__":
    """
    By default, this runs a chat demo with the model.
    To export vmfb, run with --compile_to vmfb
    To export torch, run with --compile_to torch
    """
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
        # default="meta-llama/Llama-2-7b-chat-hf", # llama causes AttributeError: 'LlamaModel' object has no attribute 'model'
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
    parser.add_argument("--quantization", type=str, default="int4", help="")
    parser.add_argument(
        "--precision", type=str, default="f16", help="dtype of model [f16, f32]"
    )

    args = parser.parse_args()
    from pprint import pprint
    print("Initializing stateless_llama chat with settings:")
    pprint(vars(args))

    model_builder = HFTransformerBuilder(
        example_input=torch.tensor([[1]], dtype=torch.int64),
        # auto_model=AutoModelForCausalLM, # if this line is missing, you may incounter "AttributeError: 'LlamaModel' object has no attribute 'model'". If this line is present, 
        # If this line is present, you may encounter
        #   File "/home/xida/miniconda/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py", line 835, in forward
        #     batch_size, seq_length = input_ids.shape[:2]
        #     ^^^^^^^^^^^^^^^^^^^^^^
        # ValueError: not enough values to unpack (expected 2, got 1)
        hf_id=args.hf_model_name,
        hf_auth_token=args.hf_auth_token,
        external_weights=args.external_weights,
        quantization=args.quantization,
        precision=args.precision,
        compile_to=args.compile_to,
    )

    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    print("Checking for stored safetensors weights...")
    # check if safetensor exists
    safetensor_path = f"{safe_name}_{args.precision}_{args.quantization}.safetensors"
    if not os.path.exists(safetensor_path):
        print("No safetensor weights found, generating...")
        from turbine_models.gen_external_params.gen_external_params import quantize
        if args.precision == "f16":
            model = model_builder.model.half()
            dtype = torch.float16
        elif args.precision == "f32":
            model = model_builder.model
            dtype = torch.float32
        else:
            sys.exit(f"{args.precision} not supported (supported precisions: f16 or f32)")
        quant_weights = quantize(model, args.quantization, dtype)
        # TODO: Add more than just safetensor support
        import safetensors
        
        safetensors.torch.save_file(quant_weights, safetensor_path)
        print("Saved safetensor weights to ", safetensor_path)
    else:
        print("Found safetensor weights, skipping generation")
    
    
    print("Checking for stored compiled vmfb module...")
    if not os.path.exists(f"{safe_name}.vmfb"):
        print("No vmfb module found, compiling...")
        compiled_module = model_builder.get_compiled_module()
        compiled_module.save_vmfb(f"{safe_name}.vmfb", target_backends=["llvm-cpu"])
        print("Saved vmfb module to ", f"{safe_name}.vmfb")
    else:
        print("Found vmfb module, skipping compilation")



    llama = StatelessLLAMALanguageModel(
        hf_model_name=args.hf_model_name,
        vmfb_path=f"{safe_name}.vmfb",
        external_weight_file=f"{safe_name}_{args.precision}_{args.quantization}.safetensors",
        hf_auth_token=args.hf_auth_token,

    )

    chat_history = "Assistant: Hi! How can I help you?\n"
    llama.initialize_generation(chat_history)
    print(chat_history, end="")

    while True:
        user_input = input("User: ")
        # quit when q, quit, or EOF is entered
        if user_input.lower() == 'quit' or user_input.lower() == 'q' or user_input == '':
            break

        chat_history = chat_history + f"User: {user_input}\nAssistant: "
        encoded_input = llama.tokenizer.encode(user_input)
        for token in tqdm(encoded_input, desc="Consuming tokens", leave=False):  # Process each token in user input
            llama.put(torch.tensor([[token]]))  # Add an extra dimension to the tensor

        response = ''
        for _ in tqdm(range(20), desc="Producing tokens", leave=False):
            token = llama.get()
            if token.item() == llama.tokenizer.eos_token_id:  # Break on end of sentence token
                break
            response += llama.tokenizer.decode(token.item())

        print(f"Assistant: {response}")

        chat_history += f"{response}\n"
