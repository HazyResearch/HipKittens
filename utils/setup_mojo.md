

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

Environment:
```
curl -fsSL https://pixi.sh/install.sh | sh

pixi init life \
  -c https://conda.modular.com/max-nightly/ -c conda-forge \
  && cd life

pixi add modular
```

Run: 
```
mojo kernel.mojo
```



