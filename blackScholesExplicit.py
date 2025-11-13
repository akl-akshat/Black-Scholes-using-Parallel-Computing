import numpy as np

#Vals of Vars
K = 85.0 #Strike Price   
T = 1.0 #Expiration Time
r = 0.1 #risk free rate
sigma = 0.2 #volatility
initial_stock_price = 70

Smax = 2 * K     
M = 80 #Price Step
N = 20000 #Time Step

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
    
stability_condition = 1 / (sigma**2 * M**2)
if dt > stability_condition:
    print(f"Warning: Stability condition may be violated.")
    print(f"dt = {dt:.6f} should be <= {stability_condition:.6f}")
    print("Consider increasing N (number of time steps).")

#actual calculation
for i in range(N - 1, -1, -1):
    for j in range(1, M):
        aj = 0.5 * dt * (sigma**2 * j**2 - r * j)
        bj = 1 - dt * (sigma**2 * j**2 + r)
        cj = 0.5 * dt * (sigma**2 * j**2 + r * j)

        V[j, i] = aj * V[j - 1, i + 1] + bj * V[j, i + 1] + cj * V[j + 1, i + 1]

#finding val of option at the initial stock price
price_index = np.searchsorted(S_vector, initial_stock_price)

if price_index < len(S_vector):
    calculated_price = V[price_index, 0]
    print(f"Calculated option price at t=0 for S={initial_stock_price}: {calculated_price:.4f}")
else:
    print(f"Initial stock price {initial_stock_price} is out of the grid's range.")