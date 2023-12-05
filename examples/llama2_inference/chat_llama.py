import argparse
import sys
from turbine_models.model_builder import HFTransformerBuilder
import torch
import iree.runtime as ireert
from transformers import AutoTokenizer
import os
import re
from tqdm import tqdm


class TurbineChatbot:
    def __init__(self, hf_model_name: str, vmfb_path: str = None, external_weight_file: str = None, hf_auth_token: str = None):
        """
        Initializes the chatbot with the given model and paths.

        :param hf_model_name: The name of the Hugging Face model.
        :param vmfb_path: Path to the .vmfb model file.
        :param external_weight_file: Path to the external weight file (optional).
        :param hf_auth_token: Hugging Face authorization token (optional).
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

    def generate_response(self, prompt: str, max_length: int = 1024, break_on_eos: bool = False) -> str:
        """
        Generates a response for the given prompt.

        :param prompt: The input prompt for the model.
        :param max_length: Maximum length of the response.
        :param break_on_eos: Whether to break the loop on end-of-sentence token.
        :return: A string representing the generated response.
        """
        initial_input = self.tokenizer(prompt, return_tensors="pt")
        example_input_id = initial_input.input_ids
        device_inputs = [ireert.asdevicearray(self.config.device, example_input_id)]

        ModuleCompiled = self.ctx.modules.state_update
        results = ModuleCompiled["run_initialize"](*device_inputs)

        tokens = []
        for _ in tqdm(range(max_length), desc="Generating tokens"):
            next_token_tensor = torch.tensor(results.to_host()[0][0].item())
            tokens.append(next_token_tensor)

            results = ModuleCompiled["run_forward"](results)

            if next_token_tensor.item() == self.tokenizer.eos_token_id and break_on_eos:
                break

        return self.tokenizer.decode(tokens)
    
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


# # Example Usage
# chatbot = TurbineChatbot("llSourcell/medllama2_7b", "medllama2_7b.vmfb", external_weight_file="medllama2_7b_f16_int4.safetensors")

# prompt = "System: Hi! How can I help you?\n"
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == 'quit':
#         break

#     full_prompt = prompt + f"User: {user_input}\nLLM:"
#     response = chatbot.generate_response(full_prompt, max_length=50, break_on_eos=True)
#     print(f"LLM: {response}")

#     prompt += f"User: {user_input}\nLLM: {response}\n"

# Example Usage



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
parser.add_argument("--quantization", type=str, default="int4", help="")
parser.add_argument(
    "--precision", type=str, default="fp16", help="dtype of model [f16, f32]"
)

args = parser.parse_args()

# first prep 

if __name__ == "__main__":
    """
    By default, this runs a chat demo with the model.
    To export vmfb, run with --compile_to vmfb
    To export torch, run with --compile_to torch
    """
    args = parser.parse_args()
    print(args.compile_to)

    model_builder = HFTransformerBuilder(
        example_input=torch.tensor([[1]], dtype=torch.int64),
        hf_id=args.hf_model_name,
        hf_auth_token=args.hf_auth_token,
        external_weights=args.external_weights,
        quantization=args.quantization,
        precision=args.precision,
        compile_to=args.compile_to,
    )
    model_builder.build_model()

    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)

    # check if safetensor exists
    safetensor_path = f"{safe_name}_{args.precision}_{args.quantization}.safetensors"
    if not os.path.exists(safetensor_path):
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
        print("Saved safetensor output to ", safetensor_path)
    
    
    # check if vmfb (compiled module) exists
    if not os.path.exists(f"{safe_name}.vmfb"):
        compiled_module = model_builder.get_compiled_module()
        compiled_module.save_vmfb(f"{safe_name}.vmfb")


    chatbot = TurbineChatbot(hf_model_name=args.hf_model_name, vmfb_path=f"{safe_name}.vmfb", external_weight_file=f"{safe_name}_{args.precision}_{args.quantization}.safetensors", hf_auth_token=args.hf_auth_token)

    # chatbot = TurbineChatbot("llSourcell/medllama2_7b", "medllama2_7b.vmfb", external_weight_file="medllama2_7b_f16_int4.safetensors")


    chat_history = "Assistant: Hi! How can I help you?\n"
    chatbot.initialize_generation(chat_history)

    while True:
        user_input = input("User: ")
        # quit when q, quit, or EOF is entered
        if user_input.lower() == 'quit' or user_input.lower() == 'q' or user_input == '':
            break

        chat_history = chat_history + f"User: {user_input}\nAssistant: "
        encoded_input = chatbot.tokenizer.encode(user_input)
        for token in tqdm(encoded_input, desc="Consuming tokens", leave=True, enable=None):  # Process each token in user input
            chatbot.put(torch.tensor([[token]]))  # Add an extra dimension to the tensor

        response = ''
        for _ in tqdm(range(20), desc="Producing tokens", leave=True, enable=None):
            token = chatbot.get()
            if token.item() == chatbot.tokenizer.eos_token_id:  # Break on end of sentence token
                break
            response += chatbot.tokenizer.decode(token.item())

        print(f"Assistant: {response}")

        chat_history += f"{response}\n"