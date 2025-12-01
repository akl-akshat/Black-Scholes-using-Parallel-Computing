# Black-Scholes-using-Parallel-Computing
🚀 Black–Scholes Option Pricing using Parallel Computing
Finite Difference Methods • MPI (Python) • OpenMP (C++) • High-Performance Computing
<div align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/7/70/Black-Scholes_Formula.png" width="500"> </div>
📌 Overview

This project implements high-performance numerical solvers for the Black–Scholes Partial Differential Equation (PDE) using:

Explicit Finite Difference Method

Parallel Distributed Computing with MPI (Python + mpi4py)

Parallel Shared-Memory Computing with OpenMP (C++)

The goal is to accelerate European option pricing using both distributed systems (MPI) and multicore CPUs (OpenMP) — comparing accuracy, stability, and computational efficiency.

This repository demonstrates end-to-end scientific computing: mathematical modelling, numerical PDEs, parallelization, and performance benchmarking.

✨ Key Features
🔢 1. Finite Difference Solvers

Explicit black–scholes PDE solver

Accurate enforcement of boundary & terminal conditions

Numerical stability handling via explicit scheme constraints

⚡ 2. MPI Parallel Solver (Python)

BSE_Explicit_Parallel.py implements:

Domain decomposition across processes

Ghost-cell communication with Sendrecv

Final solution reassembly via Gatherv

Scales efficiently across multiple CPU nodes

<div align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Mpi_logo.svg/512px-Mpi_logo.svg.png" width="120"/> </div>
🧵 3. OpenMP Parallel Solver (C++)

BSE_ExplicitParallel_OpenMP.cpp features:

Persistent OpenMP teams

Parallelized spatial loops

Cache-friendly memory access

Optional thread count via command-line arguments

<div align="center"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/OpenMP_logo.png/512px-OpenMP_logo.png" width="180"/> </div>
📊 4. Full Analysis & Presentation

The repo includes:
✔ 30+ slide research presentation
✔ Comparison sheet with runtime & accuracy data
✔ Graphs, insights, and mathematical derivations

These files highlight clear communication and presentation quality—great for interviews.

🔍 Core Concepts Implemented
🧮 Black–Scholes PDE
∂
𝑉
∂
𝑡
+
1
2
𝜎
2
𝑆
2
∂
2
𝑉
∂
𝑆
2
+
𝑟
𝑆
∂
𝑉
∂
𝑆
−
𝑟
𝑉
=
0
∂t
∂V
	​

+
2
1
	​

σ
2
S
2
∂S
2
∂
2
V
	​

+rS
∂S
∂V
	​

−rV=0
🟦 Explicit Finite Difference Discretization
𝑉
𝑗
𝑛
−
1
=
𝑎
𝑗
𝑉
𝑗
−
1
𝑛
+
𝑏
𝑗
𝑉
𝑗
𝑛
+
𝑐
𝑗
𝑉
𝑗
+
1
𝑛
V
j
n−1
	​

=a
j
	​

V
j−1
n
	​

+b
j
	​

V
j
n
	​

+c
j
	​

V
j+1
n
	​

📐 Boundary & Terminal Conditions

𝑉
(
𝑆
,
𝑇
)
=
max
⁡
(
𝑆
−
𝐾
,
0
)
V(S,T)=max(S−K,0)

𝑉
(
0
,
𝑡
)
=
0
V(0,t)=0

𝑉
(
𝑆
max
⁡
,
𝑡
)
=
𝑆
max
⁡
−
𝐾
𝑒
−
𝑟
(
𝑇
−
𝑡
)
V(S
max
	​

,t)=S
max
	​

−Ke
−r(T−t)

🚀 Performance Highlights
🏎 Runtime Summary
Method	Avg Runtime	Stability	Notes
Explicit (Serial)	Slowest	Conditionally stable	Requires very small dt
Implicit	Faster	Unconditionally stable	Solves tridiagonal system
Crank–Nicolson	Fastest + Most Accurate	Unconditionally stable	2nd order accuracy
MPI Explicit	Massive speedup	—	Distributed computation
OpenMP Explicit	10× faster locally	—	Multicore parallelism
📉 Accuracy Observations

Crank–Nicolson nearly matches analytic values

Explicit/Implicit slightly less accurate near strike

Parallel versions maintain identical accuracy
