
## Setup mi300x

Load docker:
```bash
podman run -it --privileged --network=host --ipc=host \
  -v /shared/amdgpu/home/tech_ops_amd_xqh/simran:/workdir \
  --workdir /workdir docker.io/rocm/pytorch \
  bash
```

Or:
```bash
docker run -it --privileged --network=host --ipc=host \
  -v /shared/amdgpu/home/tech_ops_amd_xqh/simran:/workdir \
  --workdir /workdir docker.io/rocm/pytorch \
  bash
```




