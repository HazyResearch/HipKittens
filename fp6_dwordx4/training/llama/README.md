
## Llama 1B training

### Setup commands 

```python
# install train extra dependencies
pip install -e .[train]
```

We breakdown this section into three parts: 1) how to set up a training config and launch; 2) how to set up fast training kernels, and 3) how to install extra optimizations for training.

### Launching training
To train a new model, construct a config.yaml file at ```train/configs/experiment/```. You can launch using the following script:
```
cd train/
CUDA_VISIBLE_DEVICES=0 python train/run.py experiment=example/llama-1b trainer.devices=1        # pytorch
CUDA_VISIBLE_DEVICES=1 python train/run.py experiment=example/llama-1b-aiter trainer.devices=1  # aiter
CUDA_VISIBLE_DEVICES=3 python train/run.py experiment=example/llama-1b-hk trainer.devices=1     # hip
```




