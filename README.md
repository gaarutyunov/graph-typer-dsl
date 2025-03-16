# GraphTyperDSL: Type Inference using Deep Similarity Learning from Code Represented as Graph

## Install dependencies

!IMPORTANT!

Only install the dependencies in a virtual environment.

Before installing the dependencies you need to [activate](https://hpc.hse.ru/instructions/base#module) the necessary modules.
The one you need has CUDA and torch already installed.
This repository uses `torch==1.11.0`. 
It is not recommended to use another version of torch, because the code might not work.
Each torch is built for a specific version of CUDA. 
You might get errors about the incompatibility between torch and CUDA versions.

To install CPU only dependecies use `install_cpu.sh`.
To install dependencies for GPU use `install.sh`.
For HSE HPC use the `install.sh` file.

## Run the scripts

More detailed information is in [this](scripts/README.md) readme file.

## Citation

```bibtex
@software{graph-typer,
  title = {{GraphTyperDSL: Type Inference using Deep Similarity Learning from Code Represented as Graph}},
  author = {German Arutyunov, Sergey Avdoshin},
  url = {https://www.github.com/gaarutyunov/graph-typer-dsl},
}
```