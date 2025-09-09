# ThunderKittens nanoGPT

This repository contains code for training GPT models with ThunderKittens CUDA kernels for NVidia H100 GPUs. We adapt the popular nanoGPT repository. 

## Setup

Create an environment and install nanoGPT dependencies:
```bash
conda create -n env python=3.12

pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Install ThunderKittens kernels:
```
git@github.com:HazyResearch/ThunderKittens.git
cd ThunderKittens/
source env.src
``` 

Select "attn" in ThunderKittens/config.py and run:
```
python setup.py install
```

## Benchmark

Let's first benchmark the kernel to make sure that everything is set up correctly. Prepare `data` of choice from NanoGPT README below (modify ``dataset`` in `bench.py` path accordingly - default = `shakespeare_char`). 
To benchmark the TK Forward Causal Attention, set `TK_kernel` = True in `bench.py` and run:


```bash
python scripts/bench.py
```

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!


## Run training from scratch

We can train a full model using our kernels:
```bash
python data/openwebtext/prepare.py # data
python train.py ./configs/ref_train_gpt2.py # train
python train.py ./configs/tk_train_gpt2.py # train
```

## Run inference with a pre-trained model

Here is a script you can use to sample from the largest available `gpt2-medium` model with and without TK kernels. 
```bash
python scripts/inference.py
```

## Citations
We build on the [nanoGPT](https://github.com/karpathy/nanoGPT) repository. To learn more about the repository, please view the original README.

