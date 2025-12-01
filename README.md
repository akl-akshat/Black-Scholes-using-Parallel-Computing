# Black-Scholes-using-Parallel-Computing
Here is the formatted content, ready to be pasted directly into your `README.md` file. I have organized it with professional Markdown formatting, proper LaTeX math rendering, and a clean visual hierarchy.

-----

# 🚀 Black–Scholes Option Pricing using Parallel Computing

### Finite Difference Methods • MPI (Python) • OpenMP (C++) • High-Performance Computing

\<div align="center"\>
\<img src="[https://upload.wikimedia.org/wikipedia/commons/7/70/Black-Scholes\_Formula.png](https://upload.wikimedia.org/wikipedia/commons/7/70/Black-Scholes_Formula.png)" width="500" alt="Black Scholes Formula"\>
<br><br>
\<img src="[https://img.shields.io/badge/Python-3.8%2B-blue?logo=python\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Python-3.8%252B-blue%3Flogo%3Dpython%26logoColor%3Dwhite)" alt="Python"\>
\<img src="[https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/C%252B%252B-17-00599C%3Flogo%3Dc%252B%252B%26logoColor%3Dwhite)" alt="C++"\>
\<img src="[https://img.shields.io/badge/MPI-mpi4py-green](https://www.google.com/search?q=https://img.shields.io/badge/MPI-mpi4py-green)" alt="MPI"\>
\<img src="[https://img.shields.io/badge/OpenMP-Parallel-red](https://www.google.com/search?q=https://img.shields.io/badge/OpenMP-Parallel-red)" alt="OpenMP"\>
\</div\>

-----

## 📌 Overview

This project implements **high-performance numerical solvers** for the Black–Scholes Partial Differential Equation (PDE). The goal is to accelerate European option pricing using both distributed systems (MPI) and multicore CPUs (OpenMP), while comparing accuracy, stability, and computational efficiency against serial implementations.

This repository demonstrates end-to-end scientific computing: **mathematical modelling, numerical PDEs, parallelization, and performance benchmarking.**

### 🛠 Technologies Used

  * **Explicit Finite Difference Method** (FDM)
  * **Parallel Distributed Computing** with MPI (Python + `mpi4py`)
  * **Parallel Shared-Memory Computing** with OpenMP (C++)

-----

## ✨ Key Features

### 🔢 1. Finite Difference Solvers

  * **Explicit Black–Scholes PDE solver:** A ground-up implementation of the explicit scheme.
  * **Boundary Conditions:** Accurate enforcement of Dirichlet boundary conditions and terminal payoff.
  * **Stability Control:** Numerical stability handling via strict timestep/spatial step constraints.

### ⚡ 2. MPI Parallel Solver (Python)

\<div align="center"\>
\<img src="[https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Mpi\_logo.svg/512px-Mpi\_logo.svg.png](https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Mpi_logo.svg/512px-Mpi_logo.svg.png)" width="100"/\>
\</div\>

The `BSE_Explicit_Parallel.py` script implements distributed memory parallelization:

  * **Domain Decomposition:** Splits the spatial grid across multiple processes.
  * **Ghost-Cell Communication:** Uses `Sendrecv` to exchange boundary data between neighboring processors.
  * **Result Aggregation:** Final solution reassembly using `Gatherv`.
  * **Scalability:** Scales efficiently across multiple CPU nodes.

### 🧵 3. OpenMP Parallel Solver (C++)

\<div align="center"\>
\<img src="[https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/OpenMP\_logo.png/512px-OpenMP\_logo.png](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/OpenMP_logo.png/512px-OpenMP_logo.png)" width="150"/\>
\</div\>

The `BSE_ExplicitParallel_OpenMP.cpp` implementation focuses on shared-memory performance:

  * **Persistent Threads:** uses persistent OpenMP teams to reduce overhead.
  * **Spatial Loop Parallelization:** `pragma omp for` directives handling the spatial grid updates.
  * **Cache Optimization:** Memory access patterns optimized for CPU cache hits.
  * **Configurable:** Thread count optional via command-line arguments.

### 📊 4. Full Analysis & Presentation

This repository is designed for educational and professional demonstration:

  * ✔ **30+ Slide Research Presentation:** Detailed walkthrough of the math and code.
  * ✔ **Comparison Sheet:** Exact runtime and accuracy data.
  * ✔ **Graphs & Derivations:** Visual insights into the solution surface and error convergence.

-----

## 🔍 Core Concepts Implemented

### 🧮 Black–Scholes PDE

The governing equation for the price of an option over time:

$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$

Where:

  * $V$: Option price
  * $S$: Underlying asset price
  * $\sigma$: Volatility
  * $r$: Risk-free interest rate

### 🟦 Explicit Finite Difference Discretization

Using backward time, centered space (BTCS) discretization to solve for the previous time step $n-1$:

$$
V^{n-1}_j = a_j V^n_{j-1} + b_j V^n_j + c_j V^n_{j+1}
$$

### 📐 Boundary & Terminal Conditions

**Terminal Condition (Call Option):**

$$
V(S, T) = \max(S-K, 0)
$$

**Boundary Conditions:**

1.  At $S=0$:
    $$V(0, t) = 0$$
2.  At $S_{max}$:
    $$V(S_{\max}, t) = S_{\max} - Ke^{-r(T-t)}$$

-----

## 🚀 Performance Highlights

The following table summarizes the performance benchmarks included in the analysis:

| Method | Avg Runtime | Stability | Notes |
| :--- | :--- | :--- | :--- |
| **Explicit (Serial)** | Slowest | Conditionally Stable | Requires very small $dt$ for stability ($dt \le dx^2/2$). |
| **Implicit** | Faster | Unconditionally Stable | Solves a tridiagonal system; computationally heavier per step but allows larger $dt$. |
| **Crank–Nicolson** | Fastest (Serial) | Unconditionally Stable | **Most Accurate.** 2nd-order accuracy in both time and space. |
| **MPI Explicit** | **Massive Speedup** | Conditionally Stable | Distributed computation allows for massive grid sizes. |
| **OpenMP Explicit** | **\~10× Faster** | Conditionally Stable | Excellent local speedup via multicore parallelism. |

### 📉 Accuracy Observations

  * **Crank–Nicolson** nearly matches analytic (exact) Black-Scholes values.
  * **Explicit/Implicit** methods are slightly less accurate near the strike price $K$ due to discretization error.
  * **Parallel versions** (MPI/OpenMP) maintain identical accuracy to the serial Explicit method (numerical precision preserved).

-----

## 👨‍💻 Usage

**Running the MPI Version:**

```bash
mpiexec -n 4 python BSE_Explicit_Parallel.py
```

**Running the OpenMP Version:**

```bash
g++ -fopenmp BSE_ExplicitParallel_OpenMP.cpp -o solver
./solver
```
