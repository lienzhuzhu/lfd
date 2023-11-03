# Problem Set 6.10
# Maximizing connections in a neural network with node budget

from scipy.optimize import minimize
import numpy as np

# Objective function
def objective(L, sign=-1.0):
    return sign * (10*(L[0]-1) + L[0]*(L[1]-1) + L[1])

# Constraints
cons = ({'type': 'eq', 'fun': lambda L: 36 - sum(L)},
        {'type': 'ineq', 'fun': lambda L: L[0]},          # L1 >= 0
        {'type': 'ineq', 'fun': lambda L: L[1]})          # L2 >= 0

# Initial guess
L0 = [1, 1]

# Bounds for L1 and L2
bnds = ((0, None), (0, None))

# Solve the problem
solution = minimize(objective, L0, args=(-1.0,), bounds=bnds, constraints=cons)

# Output the results
if solution.success:
    print("L1 = ", round(solution.x[0]))
    print("L2 = ", round(solution.x[1]))
    print("Z = ", round(-1 * solution.fun))
else:
    print("Optimization failed:", solution.message)

