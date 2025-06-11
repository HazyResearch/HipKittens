# AMD-benchmarking-harness


# Install PyTorch

Need to install special PyTorch that targets AMD

From the [PyTorch website](https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

# Get the ThunderKittens-HIP repo
```
git submodule update --init --recursive
```

# Install Hip-Python
Make sure that the version matches that of your ROCm HIP SDK version. Something like this:


```
pip install -r requirements.txt
pip install --index-url https://test.pypi.org/simple "hip-python==5.7.0.470.16"
pip install -e .
```

### Directory structure
```
kernels/
    - HiP implementations of various kernels
utils/
    - helper functions for benchmarking
```

### Kernels to benchmark
So far we care about
* Element-wise add
* Element-wise multiply
* Matrix Multiply
