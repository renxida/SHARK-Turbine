set -e

pushd ~/SHARK-Turbine
# generate vmfb with this:
python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_model_name="llSourcell/medllama2_7b" --precision=f16 --quantization=int4  --external_weights=safetensors

# generate quantized safetensors via:
python python/turbine_models/gen_external_params/gen_external_params.py --hf_model_name="llSourcell/medllama2_7b" --precision=f16 --quantization=int4

# run vmfb vs torch model:
python python/turbine_models/custom_models/stateless_llama.py --run_vmfb --hf_model_name="llSourcell/medllama2_7b" --vmfb_path=medllama2_7b.vmfb --external_weight_file=medllama2_7b_f16_int4.safetensors 

