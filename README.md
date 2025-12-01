# Black-Scholes-using-Parallel-Computing
---

# 🚀 Black–Scholes Option Pricing using Parallel Computing

### **Finite Difference Methods • MPI (Python) • OpenMP (C++) • High-Performance Computing**

<div align="center">
  <img width="1600" height="808" alt="image" src="https://github.com/user-attachments/assets/2fa25ee4-a0b0-40ac-a9a1-d211b1d0c92a" />

</div>

---

## 📌 **Overview**

This project implements **high-performance numerical solvers** for the Black–Scholes Partial Differential Equation (PDE) using:

* **Explicit Finite Difference Method**
* **Parallel Distributed Computing with MPI (Python + mpi4py)**
* **Parallel Shared-Memory Computing with OpenMP (C++)**

The goal is to **accelerate European option pricing** using both **distributed systems (MPI)** and **multicore CPUs (OpenMP)** — comparing accuracy, stability, and computational efficiency.

This repository demonstrates **end-to-end scientific computing**: mathematical modelling, numerical PDEs, parallelization, and performance benchmarking.

---

## ✨ **Key Features**

### 🔢 **1. Finite Difference Solvers**

* Explicit black–scholes PDE solver
* Accurate enforcement of boundary & terminal conditions
* Numerical stability handling via explicit scheme constraints

---

### ⚡ **2. MPI Parallel Solver (Python)**

**`BSE_Explicit_Parallel.py`** implements:

* Domain decomposition across processes
* Ghost-cell communication with `Sendrecv`
* Final solution reassembly via `Gatherv`
* Scales efficiently across multiple CPU nodes

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Mpi_logo.svg/512px-Mpi_logo.svg.png" width="120"/>
</div>

---

### 🧵 **3. OpenMP Parallel Solver (C++)**

**`BSE_ExplicitParallel_OpenMP.cpp`** features:

* Persistent OpenMP teams
* Parallelized spatial loops
* Cache-friendly memory access
* Optional thread count via command-line arguments

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/OpenMP_logo.png/512px-OpenMP_logo.png" width="180"/>
</div>

---

### 📊 **4. Full Analysis & Presentation**

The repo includes:
✔ **30+ slide research presentation**
✔ **Comparison sheet with runtime & accuracy data**
✔ **Graphs, insights, and mathematical derivations**

These files highlight clear communication and presentation quality—great for interviews.

---

## 🔍 **Core Concepts Implemented**

### 🧮 *Black–Scholes PDE*

[
\frac{\partial V}{\partial t}

* \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
* rS\frac{\partial V}{\partial S}

- rV = 0
  ]

### 🟦 Explicit Finite Difference Discretization

[
V^{n-1}*j = a_j V^n*{j-1} + b_j V^n_j + c_j V^n_{j+1}
]

### 📐 Boundary & Terminal Conditions

* (V(S, T) = \max(S-K, 0))
* (V(0, t) = 0)
* (V(S_{\max}, t) = S_{\max} - Ke^{-r(T-t)})

---

## 🚀 Performance Highlights

### 🏎 **Runtime Summary**

| Method                | Avg Runtime                 | Stability              | Notes                     |
| --------------------- | --------------------------- | ---------------------- | ------------------------- |
| **Explicit (Serial)** | Slowest                     | Conditionally stable   | Requires very small dt    |
| **Implicit**          | Faster                      | Unconditionally stable | Solves tridiagonal system |
| **Crank–Nicolson**    | **Fastest + Most Accurate** | Unconditionally stable | 2nd order accuracy        |
| **MPI Explicit**      | **Massive speedup**         | —                      | Distributed computation   |
| **OpenMP Explicit**   | **10× faster locally**      | —                      | Multicore parallelism     |

### 📉 Accuracy Observations

* Crank–Nicolson nearly matches analytic values
* Explicit/Implicit slightly less accurate near strike
* Parallel versions maintain identical accuracy

---

## 🏗 Project Structure

```
📁 Black-Scholes-using-Parallel-Computing/
│── BSE_Explicit_Parallel.py            # MPI parallel solver
│── BSE_ExplicitParallel_OpenMP.cpp     # OpenMP parallel solver
│── Black_Scholes_Presentation.pdf      # Full presentation
│── ComparisonSheet.pdf                 # Runtime & accuracy data
│── README.md
```

---

## 🛠 How to Run

### 🧵 **OpenMP Version (C++)**

```bash
g++ -O3 -fopenmp BSE_ExplicitParallel_OpenMP.cpp -o bs_omp
./bs_omp 8      # run with 8 threads
```

---

### 🌐 **MPI Version (Python)**

```bash
mpiexec -n 8 python3 BSE_Explicit_Parallel.py
```

---

## 📈 Sample Plot (From Analysis)

Option Price vs Baseline Black–Scholes:

<div align="center">
  <img src="https://raw.githubusercontent.com/akl-akshat/Black-Scholes-using-Parallel-Computing/main/assets/sample-graph.png" width="500">
</div>
<img width="1174" height="461" alt="image" src="https://github.com/user-attachments/assets/3c46e09e-44f4-48ae-b8ee-fecd7d6daf7f" />


---

## 📌 Future Improvements

* GPU Acceleration (CUDA / OpenCL)
* Adaptive meshing
* American options via penalty or PSOR
* Crank–Nicolson MPI / OpenMP hybrid version

---

## ⭐ Support & Contributions

Feel free to open issues or suggest enhancements!
If you found this helpful, ⭐ star the repo!

---

