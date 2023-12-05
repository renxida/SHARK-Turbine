import subprocess
import pytest
import os
import re

# Define the parameters for the test
model_name = "llSourcell/medllama2_7b"
compile_to = "linalg"
precision = "f16"
quantization = "int4"
external_weights = "safetensors"
save_vmfb = "True"

@pytest.fixture
def clean_up():
    # Fixture to clean up generated files after test
    yield
    safe_name = model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    # remove if exists
    if os.path.exists(f"{safe_name}.mlir"):
        os.remove(f"{safe_name}.mlir")
    if os.path.exists(f"{safe_name}.vmfb"):
        os.remove(f"{safe_name}.vmfb")


def test_stateless_llama_export_script(clean_up):
    # Construct the command to run the script
    command = [
        "python",
        "python/turbine_models/custom_models/stateless_llama_export.py",
        "--compile_to", compile_to,
        "--hf_model_name", model_name,
        "--precision", precision,
        "--quantization", quantization,
        "--external_weights", external_weights,
        "--save_vmfb", save_vmfb
    ]

    # Run the script using subprocess
    subprocess.run(command, check=True)

    # Check if the .mlir and .vmfb files are created
    safe_name = model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    assert os.path.exists(f"{safe_name}.mlir"), "MLIR file not created"
    assert os.path.exists(f"{safe_name}.vmfb"), "VMFB file not created"

    # Additional checks can be added here if needed
