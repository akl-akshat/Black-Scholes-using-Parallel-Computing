import numpy as np

#Vals of Vars
K = 85.0 #Strike Price   
T = 1.0 #Expiration Time
r = 0.1 #risk free rate
sigma = 0.2 #volatility
initial_stock_price = 70

Smax = 2 * K     
M = 80 #Price step
# Note: We can use a much smaller N compared to the explicit method because this method is unconditionally stable.
N = 800 #Time step

dS = Smax / M
dt = T / N

V = np.zeros((M + 1, N + 1)) #grid, +1 for 0th pos

S_vector = np.linspace(0, Smax, M + 1) #going from 0 to Smax in M+1 steps
t_vector = np.linspace(0, T, N + 1) #going from 0 to T in N+1 steps

#boundary condition at expiration time(t=T)
V[ : , N ] = np.maximum(S_vector - K, 0)

#boundary condition at zero stock price(S=0)
V[0 , : ] = 0

#boundary condition at high stock price(S=Smax)
for i in range(N + 1):
    V[M, i] = Smax - K * np.exp(-r * (T - t_vector[i]))
    
# This is the core of the program where we solve the matrix system A*u = b at each time step.

# Pre-calculate the coefficients for the tridiagonal matrix 'A'.
# These are constant through time.
j = np.arange(1, M) # Vector of interior j indices
aj = 0.5 * dt * (r * j - sigma**2 * j**2)
bj = 1 + dt * (r + sigma**2 * j**2)
cj = 0.5 * dt * (-r * j - sigma**2 * j**2)

# Assemble the tridiagonal matrix 'A'
# A is size (M-1) x (M-1) for our M-1 interior points.
A = np.diag(aj[1:], k=-1) + np.diag(bj) + np.diag(cj[:-1], k=1)

for i in range(N, 0, -1):
    # Construct the knowns vector b
    b = V[1:M, i].copy()

    # Adjust the 'b' vector for the known boundary conditions
    # For j=1, the term with V(0,t) moves to the right side
    b[0] -= aj[0] * V[0, i-1] # V[0, i-1] is V(S=0) at the unknown time step

    # For j=M-1, the term with V(Smax,t) moves to the right side
    b[-1] -= cj[-1] * V[M, i-1] # V[M, i-1] is V(S=Smax) at the unknown time step

    # Solve the linear system A * u = b for the unknown vector 'u'
    # 'u' will be our V[1:M, i-1], the solution at the previous time step
    u = np.linalg.solve(A, b)

    # Update the grid with the solution for the current time step
    V[1:M, i-1] = u

price_index = np.searchsorted(S_vector, initial_stock_price)

if price_index < len(S_vector):
    calculated_price = V[price_index, 0]
    print(f"Calculated option price at t=0 for S={initial_stock_price}: {calculated_price:.4f}")
else:
    print(f"Initial stock price {initial_stock_price} is out of the grid's range.")
