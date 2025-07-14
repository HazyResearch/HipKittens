
## AMD kernels in ThunderKittens

### Setup

Clone this repo and set the environment path:
```
git submodule update --init --recursive
cd ThunderKittens-HIP
source env.src
```

Options to configure your environment:
1. Use docker, following the instructions in one of:

```bash
launch_docker_mi300x.md
launch_docker_mi350x.md
```

If you want to specifically use ```mojo```:
```bash
setup_mojo.md
```


2. Setup your own environment:


Pytorch: You need to install special PyTorch that targets AMD, from the [PyTorch website](https://pytorch.org/get-started/locally/):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Load the corresponding rocm version on your gpu: 
```bash
module avail
module load rocm/6.3.3
```

Warning: If there are version mismatches, the kernel results will be incorrect. 


### Example: benchmark and write kernels


To get started benchmarking a kernel (against PyTorch):

```bash
# gemm kernel
cd kernels/TK/gemm/bf16fp32/mi325x/256_256_64_16/
make clean && make
python test_python.py
```

In this folder, we have also provided the trace output for the kernel. Download this folder to your local computer:
```bash 
cd kernels/TK/gemm/bf16fp32/mi325x/256_256_64_16/ui_output_agent_33564_dispatch_48/
```

Follow the instructions at [utils/profiling_instructions.md](https://github.com/willhu-jpg/AMD-benchmarking-harness/tree/main/utils) to setup a local tool for viewing the trace, and upload the above trace folder into this tool. The trace above was produced by running the trace collection command specified in [utils/profiling_instructions.md](https://github.com/willhu-jpg/AMD-benchmarking-harness/tree/main/utils).


Follow a similar process to write new kernel variants!


### Contact

- William Hu [willhu@stanford.edu](willhu@stanford.edu)
- Simran Arora [simran@cs.stanford.edu](simran@cs.stanford.edu)

This work is under active development. 

