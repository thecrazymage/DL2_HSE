# Graph Machine Learning

Lecturer and seminarian: [Fedor Velikonivtsev](https://www.hse.ru/org/persons/816100677/)

Recordings (in Russian): [lecture](), [seminar]().

## Annotation
In the lecture, we will explore the basics of Graph Machine Learning (Graph-ML), the tasks are being solved in this domain, and the properties of the models and architectures being used in Graph-ML. In the seminar, we will train our Graph Neural Network using modern Graph-ML frameworks.

## Installation

1. Create conda environment:

```{bash}
mamba create -n graph_ml python==3.11 -y && mamba activate graph_ml
```

2. Install torch and Graph-ML libs

```{bash}
pip install torch==2.4.0 && pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html && \
pip install torch_geometric && \
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

```

3. Install requirements (and reinstall numpy)

```{bash}
pip install -r requirements.txt && pip install 'numpy<2.0' --force-reinstall
```
