# bmpc-jax

A performant Jax implementation of [Bootstrapped Model Predictive Control (BMPC)](https://openreview.net/forum?id=i7jAYFYDcM&noteId=i6zMv7RzgX). 

## Dependencies

To install the dependencies for this project (tested on Ubuntu 22.04), run

```[bash]
pip install -U numpy tqdm "flax[all]" optax jaxtyping einops gymnasium[mujoco]==1.0.0 hydra-core tensorboard orbax-checkpoint dm_control tensorflow tensorflow-probability tf-keras
pip install -U "jax[cuda12]"
```

## Installation

Install the package from the base directory with

```[bash]
pip install -e .
```

## Usage

Then, edit ```config.yaml``` and run ```train.py``` in the main project directory. Some examples:

```[bash]
# gymnasium 
python train.py env.backend=gymnasium env.env_id=HalfCheetah-v4 
# dmcs
python train.py env.backend=dmc env.env_id=cheetah-run   
```

## Acknowledgements

Special thanks to @wertyuilife2 for their contributions to this repository!
