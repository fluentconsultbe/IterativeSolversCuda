This project aims at implementing various iterative matrix solvers on the CPU and the GPU.

It will be implemented using C++ and CUDA.

A performance comparison between the CPU and GPU will outlined performed.

# Jacobi Iterative Solvers

The jacobi iterative algorithm solved equations of the form:

Ax = b

iteratively calculating the i'th iteration vector as follows:

This algorithms is adequately suited for parallization in which the row summation

can be calculated once per thread.



