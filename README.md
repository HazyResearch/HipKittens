# HipKittens

HipKittens is a repository in the ThunderKittens cinematic universe! This work provides minimal, opinionated C++ embedded programming primitives to help you write speedy AMD AI kernels. HipKittens is built from the hardware up: we do what the silicon tells us. 

<div align="center" >
    <img src="assets/hipkittens.png" height=250 alt="HipKittens logo" style="margin-bottom:px"/> 
      <p><em>HipKittens surfing the ~wave~ (not warp).</em></p>
</div>

<br>

**AI has largely used a single hardware vendor in the past, but how can we enable a *multi-silicon* future?** Towards the dream of a single software framework that translates across hardware platforms, we explore whether the primitives used in prior DSLs (like TK) suffice for AMD, or whether we need entirely new primitives.

We find that core tile and bulk compute interfaces carry over from TK to HK, but decisions around memory access patterns, scheduling compute and memory, and ordering thread blocks within the chiplet architecture differ. HipKittens features the following types of primitives. 
1. **Tile primitives**: sized according to the tensor core units. Tile memory ops are coalesced, bank conflict free, and eagerly use tensor core layouts. We focus on minimizing address computation costs. 
2. **Python-inspired functions**: bulk compute functions that operate over tiles. These are lightweight, wrapping assembly and HIP.
3. **Asynchronous loads/stores**: hide latencies and address generation using direct buffer loads to shared memory.
4. **Scheduling and overlapping**: we show two core patterns for overlapping compute and memory, 8-wave ping pong and 4-wave interelave, that appear across kernels.

We support CDNA3 and CDNA 4. 



## Setup

```bash
# clone the repo
git clone git@github.com:HazyResearch/HipKittens.git

# obtain an amd docker using docker pull or podman pull
podman pull docker.io/rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta

# enter the docker
podman run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash

# set the environment variables
cd HipKittens/
source env.src
```

## Unit tests

We provide unit tests for you to optionally test the correctness of library functions. 

```bash
cd HipKittens/tests/unit
make -j64
```

## Quick start: running kernels

We assume you will run the following on an MI350X or MI355X unless otherwise specified. 

1. **BF16 GEMM**
```bash
# Defaults to 8192x8192x8192
# This will compare to AITER and PyTorch automatically.
cd kernels/gemm/bf16fp32/mi350x/
make clean && make
python test_python.py

# On the mi300x or mi325x run:
cd kernels/gemm/bf16fp32/mi325x/
make clean && make
python test_python.py
```

2. **Attention forwards (MHA, GQA, Causal, Non-causal, Head dim 128 / 64)**

```bash
# GQA, Non-causal, D=128, N=2048, H=64, H_KV=8, B=16:
# This will compare to AITER automatically. 
cd kernels/attn/gqa/
make clean && make
python test_python.py
```

- Modify the ```ATTN_N``` sequence length (e.g., 1024, 2048, 4096, 8192), ```ATTN_H``` query heads and ```ATTN_H_KV``` key value heads (e.g., 16 and 16 for MHA), ```ATTN_D``` head dimension (i.e., 64 or 128) in the Makefile and test_python.py file to try other settings.
- Use the same process for [gqa_causal](https://github.com/HazyResearch/HipKittens/tree/main/kernels/attn/gqa_causal).

3. **Attention backwards (MHA, GQA, Causal, Non-causal, Head dim 128 / 64)**

```bash
# GQA, Non-causal, D=128, N=8192, H=64, H_KV=8, B=16:
# This will compare to AITER automatically. 
cd kernels/attn/gqa_backwards/
make clean && make
python test_python.py 
```

- Modify the settings in the same way as stated above for forwards.
- Try [gqa_causal_backwards](https://github.com/HazyResearch/HipKittens/tree/main/kernels/attn/gqa_causal_backwards).

4. **Memory bound**

```bash
# Rotary (default B=16, H=16, D=128, N=2048)
# This will compare to AITER, PyTorch, PyTorch compiled automatically.
cd kernels/rotary/
make clean && make
python test_python.py
```

```bash
# Layernorm fused (default B=16, H=16, D=128, N=4096)
# This will compare to PyTorch, PyTorch compiled automatically.
cd kernels/layernorm/
make clean && make
python test_python.py
```

Potental issues:
- If you see a complaint that AITER is not building in the ```test_python.py``` files, then instal AITER from source [following this README.md](https://github.com/ROCm/aiter/tree/main). Luckily, it is very quick! You can also comment out AITER from ```test_python.py``` if you only need the HK kernel.
- If you see an error that ```bin/hipcc/``` is not found, then edit the Makefile to replace ROCM_BUILD_DIR with ```/opt/rocm/bin/hipcc```


## Benchmarking

Under [HipKittens/analysis](https://github.com/HazyResearch/HipKittens/tree/main/analysis) we provide scripts and instructions to benchmark all the HK kernels from our paper. This will sweep over different dimensions and settings, and we provide plotting scripts. 


## Training

Under [HipKittens/training](https://github.com/HazyResearch/HipKittens/tree/main/training) we provide instructions to train either BERT or Llama models using HipKittens attention kernels, AITER kernels, or PyTorch kernels. These are lightweight. Run them within the AMD Docker.

## Resources

We provide resources for profiling kernels, dockers, and HipKittens in [HipKittens/docs](https://github.com/HazyResearch/HipKittens/tree/main/docs). Contribute to our [onboarding documents](https://docs.google.com/document/d/15-Zvf6e0NLX1si4ml4sUOWCDlXNMtOWKiuo6CKZMEYA/edit?usp=sharing).

### Get in touch!

Contact: William Hu [willhu@stanford.edu](willhu@stanford.edu) and Simran Arora [simran@cs.stanford.edu](simran@cs.stanford.edu).
Join us on Discord to get involved, [invitation link](https://discord.com/channels/1189498204333543425/1300872762163728550)! We welcome community contributions.


