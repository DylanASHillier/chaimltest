## OPT Inference
Repo for coding challenge done at ChaiML
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
1. 
### Inference as a Flask App/ over docker instance