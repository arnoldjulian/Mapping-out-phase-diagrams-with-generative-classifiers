# Mapping out phase diagrams with generative classifiers
This repository contains a Julia implementation of the generative approach to phase-classification tasks introduced in our
[paper](https://arxiv.org/abs/2306.xxxxx).

### Abstract of the paper
One of the central tasks in many-body physics is the determination of phase diagrams which can be cast as a classification problem. Typically, classification problems are tackled using discriminative classifiers that explicitly model the conditional probability of labels given a sample. Here, we show that phase-classification problems are naturally suitable to be solved using generative classifiers that are based on probabilistic models of the measurement statistics underlying the physical system. Such a generative approach benefits from modeling concepts native to the realm of statistical and quantum physics, as well as recent advances in machine learning. This yields a powerful framework for mapping out phase diagrams of classical and quantum systems in an automated fashion capable of leveraging prior system knowledge.

![](./assets/method.png)

<p align="center">
<img src="./assets/method.png" width="50%" height="50%">
</p>

### This repository

contains code to map out phase diagrams given generative models. The source files can be found in [source folder](./src/). We provide exemplary code for

* the equilibrium phase diagram of the two-dimensional anisotropic Ising model (of size 20 x 20), see [the folder](./examples/Ising/).

The corresponding data can be found in the [data folder](./data/). Other physical systems can be analyzed in the same fashion.

### How to run / prerequisites:

- install [julia](https://julialang.org/downloads/)
- download, `activate`, and `instantiate` [`Pkg.instantiate()`] our package
- individual files can then be executed by calling, e.g., `julia run_main.jl`
- uncomment `savefig()` functions to save plots

## Authors:

- [Julian Arnold](https://github.com/arnoldjulian)
- [Frank Schäfer](https://github.com/frankschae)
- Alan Edelman
- Christoph Bruder

```
@article{arnold:2023,
  title={Mapping out phase diagrams with generative classifiers},
  author={Arnold, Julian and Schäfer, Frank and Edelman, Alan and Bruder, Christoph},
  journal={arXiv:2305:xxxxx},
  year={2023},
  url = {https://arxiv.org/abs/2305:xxxxx}
}
```
