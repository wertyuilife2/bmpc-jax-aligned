from setuptools import setup, find_packages

setup(
    name='bmpc-jax',
    version='0.1.0',    
    description='Jax implementation of BMPC',
    url='https://github.com/ShaneFlandermeyer/tdmpc2-jax',
    author='Shane Flandermeyer',
    author_email='shaneflandermeyer@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
      'jax',
      'jaxlib',
      'numpy',
      'tqdm',
      'flax[all]',
      'optax',
      'jaxtyping',
      'einops',
      'gymnasium[mujoco]',
      'hydra-core',
      'tensorboard',
      'orbax-checkpoint',
      'dm_control',
      'tensorflow',
      'tensorflow-probability',
      'tf-keras'
    ],

)