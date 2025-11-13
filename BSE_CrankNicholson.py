import numpy as np
import matplotlib.pyplot as plt

K = 85.0
T = 1.0
r = 0.10
sigma = 0.20

S_max = 2 * K
M = 80          # price steps (grid points M+1)
N = 100         # time steps

dS = S_max / M
dt = T / N

# Grid
S = np.linspace(0, S_max, M + 1)      # S[0..M]
t_vec = np.linspace(0, T, N + 1)      # t[0..N]

# Option value grid V[j, n] where j=0..M, n=0..N
V = np.zeros((M + 1, N + 1))

# Final payoff at expiry (t = T) -> last time column index N
V[:, N] = np.maximum(S - K, 0)

# Boundary conditions for all times
V[0, :] = 0.0
# V(S_max, t) = S_max - K*exp(-r*(T - t))
for n in range(N + 1):
    tau = t_vec[n]
    V[M, n] = S_max - K * np.exp(-r * (T - tau))

# Interior indices j = 1..M-1
j = np.arange(1, M)

# Build per-j spatial operator coefficients (L operator: a_j u_{j-1} + b_j u_j + c_j u_{j+1})
# Derived using central differences with S_j = j*dS.
alpha = 0.5 * (sigma**2 * j**2 - r * j)      # coefficient multiplying u_{j-1}
beta  = - (sigma**2 * j**2 + r)            # coefficient multiplying u_j
gamma = 0.5 * (sigma**2 * j**2 + r * j)      # coefficient multiplying u_{j+1}

# Build matrices A and (we will use B via vectorized expression)
# A = I - 0.5*dt*L  where L is tri-diagonal given by (alpha,beta,gamma)
# The unknowns vector u corresponds to V[1:M, n-1] (interior points)
m = M - 1   # number of interior points

diag_A = (1.0 - 0.5 * dt * beta)            # main diagonal of A (length m)
lower_A = -0.5 * dt * alpha[1:]             # sub-diagonal (length m-1): corresponds to alpha[1..]
upper_A = -0.5 * dt * gamma[:-1]            # super-diagonal (length m-1): corresponds to gamma[0..m-2]

# Assemble A as a dense array (OK for moderate M). For large M use sparse.
A = np.diag(diag_A) + np.diag(lower_A, k=-1) + np.diag(upper_A, k=1)

# Time-stepping backward (Crank-Nicolson)
for n in range(N, 0, -1):
    # V at time level n (known)
    Vn = V[:, n]

    # Build right-hand side b = (I + 0.5*dt*L) * Vn (for interior points j=1..M-1)
    # Component-wise:
    b = (1.0 + 0.5 * dt * beta) * Vn[1:M] \
        + 0.5 * dt * alpha * Vn[0:M-1] \
        + 0.5 * dt * gamma * Vn[2:M+1]

    # The term for j=1 (b[0]) involving V[0, n-1] is 0 due to the
    # boundary condition, so we only need to add the term for j=M-1 (b[-1])
    # involving V[M, n-1].
    b[-1] = b[-1] + (0.5 * dt * gamma[-1] * V[M, n-1])  

    # Solve A * u = b where u is interior V at time n-1
    u = np.linalg.solve(A, b)

    V[1:M, n-1] = u

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(S, V[:, 0], label='Option price t=0 (Crank–Nicolson)')
plt.plot(S, V[:, N], '--', label='Payoff at expiry')
plt.xlabel('Stock price S')
plt.ylabel('Option price V')
plt.title('European Call: Crank–Nicolson FDM')
plt.legend()
plt.grid(True)
plt.show()

# Retrieve / interpolate price for a specific spot
initial_stock_price = 70.0
calculated_price = np.interp(initial_stock_price, S, V[:, 0])

print(f"\nParameters:")
print(f"  K = {K}, T = {T}, r = {r}, sigma = {sigma}")
print(f"  M = {M}, N = {N}")
print("-" * 40)
print(f"Calculated option price at t=0 for S={initial_stock_price}: {calculated_price:.6f}")