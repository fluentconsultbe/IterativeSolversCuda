This project focuses on implementing and comparing several iterative matrix solvers on both the CPU and the GPU. 

The implementation is written in **C++** for the CPU components and **CUDA** for the GPU‑accelerated versions.

A performance comparison between CPU and GPU implementations is carried out to highlight the benefits and trade‑offs of GPU acceleration for iterative linear solvers.

# Jacobi Iterative Solvers

The Jacobi method solves linear systems of the form:

$$A x = b$$

by iteratively computing the next approximation $x^{(k+1)}$ from the previous one $x^{(k)}$.  

Each component is updated independently:

$$x_i^{(k+1)} = \frac{1}{A_{ii}} \left( b_i - \sum_{j \ne i} A_{ij} x_j^{(k)} \right)$$

This independence makes the Jacobi method **highly parallelizable**.  

Each row update can be computed by a separate thread, making it well‑suited for GPU execution.


