Code is taken from Chris Pederson the original owner of this repo. Modyfied for hpc singularity use.
Requirements are a mess right now as we are using jax_cfd for Kolmogorov flow and torch_qg for quasi-geostrophic flows, so this is not yet ready for public consumption.

jax_cfd: https://github.com/google/jax-cfd
torch_qg: https://github.com/Chris-Pedersen/torch_qg

To install in developer mode:
```pip install -e .```

To install otherwise:
```pip install . ```
