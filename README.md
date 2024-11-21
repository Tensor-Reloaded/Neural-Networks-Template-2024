![image_clipdrop-enhance](https://github.com/Tensor-Reloaded/Advanced-Topics-in-Neural-Networks-Template-2023/assets/8055539/5965f7aa-34ad-4899-b2af-be3cc084cb96)

# [Neural-Networks-2024](https://sites.google.com/view/rbenchea/neural-networks)

Repository for the Neural Networks laboratory, "Alexandru Ioan Cuza" University, Faculty of Computer Science, Bachelor degree.

## Environment setup

Google Colab: PyTorch, Pandas, and Numpy are already available.  

Local instalation: 
1. Create a Python environment (using conda or venv). We recommend installing conda from [Miniforge](https://github.com/conda-forge/miniforge).
```
# Create the environment
conda create -n 312 -c conda-forge python=3.12
# activate the environment
conda activate 312
# Run this to use conda-forge as your highest priority channel (not needed if you installed conda from Miniforge)
conda config --add channels conda-forge
```
2. Install PyTorch 2.4.1+ from [pytorch.org](https://pytorch.org/get-started/locally/) using `conda` or `pip`, depending on your environment. 
    * Choose the Stable Release, choose your OS, select Conda or Pip and your compute platform. For Linux and Windows, CUDA or CPU builds are available, while for Mac, only builds with CPU and MPS acceleration.
    * Example CPU: ```conda install pytorch torchvision torchaudio cpuonly -c pytorch```.


## Recommended resources:

- Linear algebra:
   * [Essence of linear algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (linear transformations; matrix multiplication)
   * [Essence of calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) (derivatives; chain rule)
- Backpropagation:
   * [Neural Networks (chapter 1 - chapter 4)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (animated introduction to neural networks and backpropagation)
- If you want to learn more in advance, check our [other recommended resources](https://github.com/Tensor-Reloaded/Advanced-Topics-in-Neural-Networks-Template-2024/blob/main/Resources.md).

## Table of contents

* [Lab01](./Lab01) (Homework 1: Solve linear system)
* [Lab02](./Lab02) (Homework 2: Perceptron implementation)
* [Lab03](./Lab03) (Homework 3: Multilayer Perceptron implementation)
* [Lab04](./Lab04)
* [Lab06](./Lab06) PyTorch Tensors, Autograd
* [Lab09](./Lab09) Datasets, DataLoaders (Homework 4: PyTorch Pipeline)


