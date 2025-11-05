

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

Run: 
```
mojo kernel.mojo
```



