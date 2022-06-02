## OPT Inference
Repo for coding challenge done for a job interview.

Involved doing inference on the new OPT models and trying to make the inference better/more efficient

### Model Generation
For e.g. 13b run:
1. `$ mkdir 13b`
2. `$ cd 13b`
3. `$ python3 ../quantization_tools.py --specific_model=opt-13b --optimize --quantize`
4. `$ cd ..`

### Benchmarking
Again for 13b you can run
1. `$ python3 benchmark.py --model_version=13b --num_samples=10`

### Inference with CLI
1. Setup the app_config.json: optimization_level can be "baseline", "onnx", "onnxruntime", "fusion", or "quantized". opt_version varies over the different model versions: "125m", "250m", 
1. `$ python3 app.py`

### Inference as a Flask App/ over docker instance
Not yet properly implemented...

### Important
Additionally had to change one line in optimum/onnxruntime namely adding DisableShapeInference to the extra options
