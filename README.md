# Univariate Radial Basis Function Layers: Brain-inspired Deep Neural Layers for Low-Dimensional Inputs

A novel layer based on univariate radial basis functions to improve learning for low-dimensional inputs.
Abstract

Daniel Jost, Basavasagar Patil, Xavier Alameda-Pineda, Chris Reinke

[`Paper`](https://arxiv.org/abs/2311.16148)


### Abstract
Deep Neural Networks (DNNs) became the standard tool for function approximation with most of the introduced architectures being developed for high-dimensional input data. However, many real-world problems have low-dimensional inputs for which the standard Multi-Layer Perceptron (MLP) are a common choice. An investigation into specialized architectures is missing. We propose a novel DNN input layer called Univariate Radial Basis Function (U-RBF) Layer as an alternative. Similar to sensory neurons in the brain, the U-RBF Layer processes each individual input dimension with a population of neurons whose activations depend on different preferred input values. We verify its effectiveness compared to MLPs and other state-of-the-art methods in low-dimensional function regression tasks. The results show that the U-RBF Layer is especially advantageous when the target function is low-dimensional and high-frequent.


### Installation
Create a python environment
```
conda create --name urbf python=3.11
conda activate urbf
```

Install dependencies
```
pip install experiment-utilities

pip install -e ./code/function_regression
pip install -e ./code/urbf_layer
```

Setup the Jupyter Notebook extensions
```
jupyter contrib nbextension install --user
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix qgrid
```


### Overview
- 'code' folder contains the local code packages
- 'experiments' folder contains the conducted experiments 
  - can be run using 'sh run_experiments.sh' inside the specific experiment folder
  - the jupyter notebook in the 'analyze' subfolder can be used to visualize and inspect the results 