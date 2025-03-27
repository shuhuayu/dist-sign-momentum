# Introduction

This repository contains the implementations for the distributed sign momentum algorithm with local steps (DSM-LS) in our paper: [Distributed Sign Momentum with Local Steps for Training Transformers](https://arxiv.org/abs/2411.17866). We implemented our algorithm in the [fairscale](https://github.com/facebookresearch/fairscale) framework, and the global optimizer is distributed across GPUs for memory efficiency. You can install the modified fairscale from source. 
```python
git clone git@github.com:shuhuayu/dist-sign-momentum.git
cd fairscale
pip install -e .
```
The usage is simple. Here is an example. For any base optimizer and model you like, you can use
```python
from fairscale.experimental.nn.data_parallel import FastdtDistributedDataParallel as fastdt

# you can configure your model with global step parameters of distributed training, here is an example
model = fastdt(model, # your own model
    nprocs_per_node=8, # each node has 8 gpus
    num_updates=0, # init from 0
    fastdt_frequency=12, # communication interval
    fastdt_lr=1.0, # global step size
    fastdt_use_lion=True, # use sign momentum in the global step
    fastdt_lion_beta1=0.95, 
    fastdt_lion_beta2=0.98,
    fastdt_lion_weight_decay=0.1
    )
# each node performs a local step
optimizer.step()
# cooridating global steps and gpu commnications
model.perform_fastdt(optimizer)
# zero grad
optimizer.zero_grad(set_to_none=True)
```

We also built examples using codebase [nanoGPT](https://github.com/karpathy/nanoGPT). Examples are provided in scripts:
```bash
nanoGPT/run_example.sh
```

Our experiments show that dist-sign-momentum consistently outperforms the built-in [SlowMo](https://arxiv.org/abs/1910.00643) method (with the same momery cost) in distributed training of GPT-2 Transformer models with multiple local steps. This makes our algorithm particularly well-suited for scenarios where communication is prohibitive and reducing synchronization frequency is essential. This table shows the improvement from our method compared to SlowMo in terms of validation losses and transcribed perpelexity. 

<p align="center">
  <img width="717" alt="image" src="https://github.com/user-attachments/assets/f8b4aa9f-9abc-4135-98de-8540a2390b26" />
</p>
