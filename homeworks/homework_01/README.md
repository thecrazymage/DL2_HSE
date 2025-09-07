# Homework 01 - Tensor and DL Libraries

> **Date of issue:** <span style="color:red">September 8</span>.
>
> **Deadline:** <span style="color:red">September 30, 23:59</span>.

The goal of this homework is to practice using PyTorch and gain experience with profiling and running multi-GPU training jobs in a near real-world setup.

Table of Contents:

1. [MiniTorch](#1-minitorch-make-your-own-pytorch)
2. [Tensor Puzzles](#2-tensor-puzzles-broadcasting-practice)
3. [Efficient Pytorch](#3-efficient-pytorch-speedup-challenge) 
4. [Grading info](#tldr.-Grading-and-Deadlines)

## 1. MiniTorch: Make your own PyTorch

**Goal:** Build a tensor library with automatic differentiation from scratch to understand modern DL frameworks better. Compared to the task from
[DL-1](https://github.com/xiyori/intro-to-dl-hse/tree/2024-2025/homeworks-small/shw-01-mlp) this one is more general (e.g. the library would allow you to write and differentiate any computational graph, not just a sequential MLP). It mimics the structure of a modern tensor library with autograd like PyTorch closer.

**Source:** [minitorch.github.io](https://minitorch.github.io)

**Helpful Resources**:
- https://github.com/karpathy/micrograd 
- https://colah.github.io/posts/2015-08-Backprop

**Less Helpfull, but Fun**:
- https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html 
- https://github.com/tinygrad/teenygrad
- https://huggingface.co/blog/joey00072/microjax
- https://arogozhnikov.github.io/2023/12/28/fastest-autograd.html

**Grading in Anytask**: 
Link to your completed minitorch module repository where all tests are passing in the CI (GitHub actions)
- **2 pts** — [Module 0:](https://minitorch.github.io/module0/module0/) basic intro to the test system and coding practices.
- **2 pts** - [Module 1:](https://minitorch.github.io/module1/module1/) scalar automatic differentiation system.
- **+2 pts (bonus)** - [Module 2:](https://minitorch.github.io/module2/module2/) tensor version of the library
- **+2 pts (bonus)** - [Module 3:](https://minitorch.github.io/module3/module3/) efficiency improvements


## 2. Tensor Puzzles: Broadcasting Practice

**Goal:** Master tensor operations and broadcasting by solving puzzles using only basic functions. In the previous task we've built a simple tensor programming library with basic ops. In this task we'll see how we can do many non-trivial computations using only the basic operations like `arange`, indexing and broadcasting. We'll practice using broadcasting and tensor operations to implement.

**Source:** [original notebook](https://github.com/srush/Tensor-Puzzles)

Tensor Puzzles Notebook: \
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/Tensor-Puzzles/blob/main/Tensor%20Puzzlers.ipynb)

**Grading in Anytask**: 
- **2pts** - For the code with solved puzzles, that is passing all the tests (add notebook to anytask)
- Each puzzle is equal in terms of point distribution if you solve fewer than 21 puzzles.


## 3. Efficient PyTorch: Speedup Challenge

**Goal:**

Writing efficient code is very important in DL. However, often you need to read and optimize esoteric PyTorch (or else) code. This task challenges you to optimize a slow, complex neural network for tabular data, a common issue when working with research code.

You are given a script for a slightly modified neural network for tabular data from an ICML [paper](https://arxiv.org/abs/2305.18446) (don't think too much of it, you'll see that the model is complicated and slow - perfect candidate to practice code optimization).

The [`train.py`](/homeworks/homework_01/train.py) file correctly implements the model from the paper (check the model figure in paper), the only modification is in the expansion block where we thrown away all the non-linearities. The code currently runs at ~14 samples/sec on one Kaggle T4 GPU. Your goal is to achieve a 500x speedup to ~7K samples/sec. All work must be done and benchmarked using **two T4 GPUs in a Kaggle notebook**.

**Optimization Plan**:
1. Profile the code (using PyTorch profiler, torch.utils.benchmark, print statements for your eyes)
2. Rewrite and improve the bottlenecks you've identified (e.g. rewrite some op's, without changing the computations - e.g. your modifications should not alter the model architecture or hyperparameters, only the computationally equivalent operations are allowed).
3. Use all of the efficiency gains that PyTorch provides (see helpful resources below for hints)
4. Document the process (gist, pdf, comment directly in anytask)

**Helpful Resources**:
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs
- https://pytorch.org/docs/stable/amp.html
- https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- https://sankalp.bearblog.dev/einsum-new

**Grading in Anytask**:
- **2pts** - For the task for the ~100x speed-up (rewrite some ops) (from part 1 and 2 above)
- **2pts** - For the addtional ~4-5x speed up (through multi-gpu training, amp, compile)


## Final Grading

**Grades**:

- **4pts** for the MiniTorch (+ up-to **4pts** bonus). Proof: link to repos in anytask
- **2pts** for the Tensor Puzzles. Proof: notebook with finished puzzles and passed tests.
- **4pts** for efficient PyTorch. Proof: fast script and short report on what you've changed.

For the total of **10pts** with up to **4pts** bonus.

**Materials and Cheating**:

You can use anything from the resources linked above. If you use any additional materials (e.g tutorial posts, code examples) include references to your anytask report.

If you've used any LLM system, include your prompts/questions in additon to model answers (e.g. it's ok to use LLMs, just as any reference materials from the Internet, just be transparent about it).