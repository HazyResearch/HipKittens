

### Setup mojo

Docker:
```
podman run -it --privileged --network=host --ipc=host \
  -v /shared/amdgpu/home/tech_ops_amd_xqh/simran:/workdir \
  --workdir /workdir \
  --device /dev/kfd \
  --device /dev/dri \
  --entrypoint /bin/bash \
  docker.io/modular/max-amd:nightly
```

Environment (https://docs.modular.com/mojo/manual/get-started/):
```
# if you don't have it, install pixi
curl -fsSL https://pixi.sh/install.sh | sh

# create a project
pixi init life \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd life

# install the modular conda package
pixi add modular

# setup the VM
pixi shell
```

Run benchmark: 
```
mojo test_kernel.mojo
```

## GEMM Results:
| Kernel (dim)        | met (ms)           | iters | Arithmetic (GFLOPS/s) |
| ----------- | ------------------ | ----- | --------------------- |
| Mojo matmul (1024) | 0.015481722        | 1000  | 138710.9036062009     |
| Mojo matmul (2048)| 0.038945884        | 1000  | 441121.56201153377    |
| Mojo matmul (4096)| 0.25926358         | 1000  | 530112.8429685342     |
| Mojo matmul (8192) | 1.4052444756820877 | 843   | 782434.4068261225     |
| Mojo matmul (16384) | 12.003901405940592 | 101   | 732769.5159054635     |


## MHA Results

Slow :( 
Also it's non-causal so fewer flops than reported?

| name                                  | met (ms)           | iters | Arithmetic (GFLOPS/s) |
| ------------------------------------- | ------------------ | ----- | --------------------- |
| AMD_FlashAttention_2048x2048_16h_128d | 4.4123881137724545 | 501   | 124593.71200190636    |

