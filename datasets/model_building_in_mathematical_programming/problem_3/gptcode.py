
import json
from gurobipy import *

# Read Data
with open("data.json", "r") as read_file:
    data = json.load(read_file)

# Create a new model
model = Model("Factory Optimization")

K = len(data['profit'])  # number of products
I = len(data['maintain'])  # number of months
M = len(data['num_machines'])  # number of machines

# Create Variables
sell = model.addVars(K, I, name="sell")
manufacture = model.addVars(K, I, name="manufacture")
storage = model.addVars(K, I, name="storage")

# Set Objective
model.setObjective(quicksum(data['profit'][k] * sell[k, i] - data['store_price'] * storage[k, i]
                            for k in range(K) for i in range(I)), GRB.MAXIMIZE)

# Add Constraints
# Manufacturing limit due to machine maintenance
for m in range(M):
    for i in range(I):
        model.addConstr(
            quicksum(data['time'][k][m] * manufacture[k, i] for k in range(K)) <= data['n_workhours'] * 24 *
            (data['num_machines'][m] - data['maintain'][i][m]), f"Machine_{m}_Month_{i}")

# Marketing limitations
for k in range(K):
    for i in range(I):
        model.addConstr(sell[k, i] <= data['limit'][k][i], f"Product_{k}_Month_{i}")

# Storage capacity limit
for k in range(K):
    for i in range(I):
        model.addConstr(storage[k, i] <= 100, f"Storage_{k}_Month_{i}")

# Flow balance constraints
for k in range(K):
    for i in range(I):
        if i == 0:
            model.addConstr(manufacture[k, i] == sell[k, i] + storage[k, i], f"Balance_{k}_Month_{i}")
        else:
            model.addConstr(storage[k, i-1] + manufacture[k, i] == sell[k, i] + storage[k, i], f"Balance_{k}_Month_{i}")

# Final inventory requirement
for k in range(K):
    model.addConstr(storage[k, I-1] >= data['keep_quantity'], f"Final_{k}")

# Solve model
model.optimize()

# Save output
output = {
    "sell": [[sell[k, i].x for k in range(K)] for i in range(I)],
    "manufacture": [[manufacture[k, i].x for k in range(K)] for i in range(I)],
    "storage": [[storage[k, i].x for k in range(K)] for i in range(I)]
}

with open("output.json", "w") as write_file:
    json.dump(output, write_file, indent=4)
