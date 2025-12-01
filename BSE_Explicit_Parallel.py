# mpiexec -n 8 python3 BSE_Explicit_Parallel.py

from mpi4py import MPI
import numpy as np
import time

# ---------- Problem parameters (tweak for demo)
K = 85.0
T = 1.0
r = 0.1
sigma = 0.2
initial_stock_price = 70.0
 
Smax = 2 * K
# M = 100      # number of S steps -> grid points 0..M
# N = 130000    # number of time steps (reduce for quick tests)

M = 200
N = 10000

dS = Smax / M
dt = T / N

start_time = time.perf_counter()

# ---------- MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Build S grid locally (cheap)
S = np.linspace(0.0, Smax, M + 1)
t = np.linspace(0.0, T, N + 1)

# Interior indices are j = 1 .. M-1 (these are the points we update)
global_interior_start = 1
global_interior_end = M - 1
global_interior_count = max(0, global_interior_end - global_interior_start + 1)

# Partition the interior count among processes (balanced)
def local_slice(count, size, rank):
    base, rem = divmod(count, size)
    if rank < rem:
        local_count = base + 1
        offset = rank * local_count
    else:
        local_count = base
        offset = rem * (base + 1) + (rank - rem) * base
    if local_count == 0:
        return 0, 0
    local_start = global_interior_start + offset
    return local_start, local_count

start_j, local_count = local_slice(global_interior_count, size, rank)
end_j = start_j + local_count - 1 if local_count > 0 else -1

# If this rank has work, allocate two small arrays with ghost cells:
# layout: arr[0] = left ghost (j = start_j - 1), arr[1..local_count] = actual, arr[-1] = right ghost (j = end_j + 1)
if local_count > 0:
    next_level = np.zeros(local_count + 2, dtype=np.float64)  # holds V[:, i+1]
    curr_level = np.zeros_like(next_level)
    # initialize payoff at t = T for these local S points
    local_S = S[start_j:end_j + 1]
    next_level[1:1 + local_count] = np.maximum(local_S - K, 0.0)
else:
    next_level = np.empty(0, dtype=np.float64)
    curr_level = np.empty(0, dtype=np.float64)

# Right boundary analytic value function
def V_right(time_val):
    return Smax - K * np.exp(-r * (T - time_val))

# Stability quick check
if rank == 0:
    stab = 1.0 / (sigma**2 * M**2)
    if dt > stab:
        print(f"[WARN] dt ({dt:.2e}) may violate stability <= {stab:.2e}")

# coeffs function uses global j
def coeffs(j):
    a = 0.5 * dt * (sigma**2 * j**2 - r * j)
    b = 1.0 - dt * (sigma**2 * j**2 + r)
    c = 0.5 * dt * (sigma**2 * j**2 + r * j)
    return a, b, c

# Main time loop (backwards)
for i in range(N - 1, -1, -1):
    left = rank - 1 if rank - 1 >= 0 else MPI.PROC_NULL
    right = rank + 1 if rank + 1 < size else MPI.PROC_NULL

    # Prepare scalars to send (wrap as 1-element arrays)
    if local_count > 0:
        # FIX: Wrap value in [ ] to create a 1-element (1D) array, not a 0-D scalar
        send_left = np.array([next_level[1]], dtype=np.float64)      # value at global j = start_j
        send_right = np.array([next_level[local_count]], dtype=np.float64)  # value at global j = end_j
    else:
        # FIX: Wrap value in [ ] to create a 1-element (1D) array
        send_left = np.array([0.0], dtype=np.float64)
        send_right = np.array([0.0], dtype=np.float64)

    # FIX: Initialize receive buffers as 1-element (1D) arrays
    # Using np.empty(1) is standard for receive buffers.
    recv_from_left = np.empty(1, dtype=np.float64)  # will hold neighbor's right-border
    recv_from_right = np.empty(1, dtype=np.float64) # will hold neighbor's left-border

    # Exchange: send_right -> right neighbor, receive left neighbor's right-border into recv_from_left
    comm.Sendrecv(sendbuf=send_right, dest=right, sendtag=10,
                  recvbuf=recv_from_left, source=left, recvtag=10)

    # Exchange: send_left -> left neighbor, receive right neighbor's left-border into recv_from_right
    comm.Sendrecv(sendbuf=send_left, dest=left, sendtag=11,
                  recvbuf=recv_from_right, source=right, recvtag=11)

    # Fill ghosts or boundary formulas
    if local_count > 0:
        if left == MPI.PROC_NULL:
            next_level[0] = 0.0  # global j=0 boundary
        else:
            # This now works, as recv_from_left is a 1-D array
            next_level[0] = recv_from_left[0]

        if right == MPI.PROC_NULL:
            next_level[-1] = V_right((i + 1) * dt)  # value at next time level
        else:
            # This now works, as recv_from_right is a 1-D array
            next_level[-1] = recv_from_right[0]

        # Compute local interior values for time i (using next_level which holds i+1)
        for pos in range(1, 1 + local_count):
            j_global = start_j + (pos - 1)
            a, b, c = coeffs(j_global)
            left_val = next_level[pos - 1]
            center_val = next_level[pos]
            right_val = next_level[pos + 1]
            curr_level[pos] = a * left_val + b * center_val + c * right_val

        # swap levels
        next_level, curr_level = curr_level, next_level
        # Clear curr_level (which is now next_level's old memory) for the next iteration
        # Note: Only need to clear the interior, ghosts are overwritten
        curr_level[1:1 + local_count] = 0.0
    else:
        # ranks with zero work do nothing but keep collective behavior correct
        pass

# Gather interior pieces at root
sendbuf = next_level[1:1 + local_count].copy() if local_count > 0 else np.empty(0, dtype=np.float64)
if rank == 0:
    counts = np.empty(size, dtype=np.intc) # Use np.intc for mpi4py counts
    for r in range(size):
        _, cnt = local_slice(global_interior_count, size, r)
        counts[r] = cnt
    displs = np.zeros(size, dtype=np.intc) # Use np.intc for mpi4py displacements
    displs[0] = 0
    for r in range(1, size):
        displs[r] = displs[r - 1] + counts[r - 1]
    recvbuf = np.empty(global_interior_count, dtype=np.float64)
else:
    counts = None
    displs = None
    recvbuf = None

comm.Gatherv(sendbuf, (recvbuf, counts, displs, MPI.DOUBLE), root=0)

# Root reconstructs V at t=0 and prints option price
if rank == 0:
    V0 = np.zeros(M + 1, dtype=np.float64)
    V0[0] = 0.0
    V0[M] = V_right(0.0)
    if global_interior_count > 0:
        V0[1:M] = recvbuf
    
    # Find the index closest to the initial_stock_price
    idx = np.searchsorted(S, initial_stock_price)
    # Handle case where price is off the grid or exactly on a point
    if idx > 0 and (idx == M + 1 or np.abs(S[idx] - initial_stock_price) > np.abs(S[idx-1] - initial_stock_price)):
        idx = idx - 1
        
    idx = min(idx, M) # Ensure it's within bounds
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
        
    print(f"Processes: {size}, M={M}, N={N}")
    print(f"\nParameters:")
    print(f"  K = {K}, T = {T}, r = {r}, sigma = {sigma}")
    print(f"  M = {M}, N = {N}")
    print("-" * 40)
    print(f"Option price at t=0 for S~{S[idx]:.2f} (requested {initial_stock_price}): {V0[idx]:.6f}")
    print(f"Calculation took {elapsed_time:.6f} seconds.")