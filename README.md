# ♨️ Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos ♨️ 

Code for [Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos](https://arxiv.org/abs/2503.18731) published at ICML 2025

To generate training data, we use [jax_cfd](https://github.com/google/jax-cfd) for Kolmogorov flow and [torch_qg](https://github.com/Chris-Pedersen/torch_qg) for quasi-geostrophic flows.

To install in developer mode:
```pip install -e .```

To install otherwise:
```pip install . ```

The codebase is heavily integrated with [wandb](https://wandb.ai/): for each training and inference run, a new wandb experiment is created, and all config metadata and results plots are uploaded.

The training loop for the stochastic residual emulator is [here](https://github.com/Chris-Pedersen/thermalizer/blob/ab2842df918d40e5fb3de28466ad7c1a710e5581/thermalizer/systems/training_systems.py#L180).
The core training algorithm for the thermalizer is found [here](https://github.com/Chris-Pedersen/thermalizer/blob/ab2842df918d40e5fb3de28466ad7c1a710e5581/thermalizer/systems/training_systems.py#L739).
and here are points to inference algorithms for thermalized [Kolmogorov](https://github.com/Chris-Pedersen/thermalizer/blob/main/thermalizer/kolmogorov/inference.py) and [QG](https://github.com/Chris-Pedersen/thermalizer/blob/main/thermalizer/qg/inference.py) flows.

Please cite as:

```
@article{pedersen2025thermalizer,
  doi = {10.48550/ARXIV.2503.18731},
  url = {https://arxiv.org/abs/2503.18731},
  author = {Pedersen,  Christian and Zanna,  Laure and Bruna,  Joan},
  title = {Thermalizer: Stable autoregressive neural emulation of spatiotemporal chaos},
  publisher = {arXiv},
  year = {2025},
}
```
