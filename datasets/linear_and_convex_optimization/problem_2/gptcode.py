
import json
from gurobipy import Model, GRB

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

time_required = data['time_required']
machine_costs = data['machine_costs']
availability = data['availability']
prices = data['prices']
min_batches = data['min_batches']

# Indices for parts and machines
P = range(len(prices))
M = range(len(machine_costs))

# Create a new model
model = Model('auto_parts_manufacturer')

# Add variables
x = model.addVars(P, vtype=GRB.INTEGER, name="batches")

# Set objective function
model.setObjective(sum((prices[p] - sum(machine_costs[m] * time_required[m][p] for m in M)) * x[p] for p in P), GRB.MAXIMIZE)

# Add machine availability constraints
for m in M:
    model.addConstr(sum(time_required[m][p] * x[p] for p in P) <= availability[m], name=f"Machine_{m}_availability")

# Add minimum production constraints
for p in P:
    model.addConstr(x[p] >= min_batches[p], name=f"Min_batches_{p}")

# Optimize model
model.optimize()

# Save the results
result = {
    "batches": [int(x[p].x) for p in P],
    "total_profit": model.objVal
}

with open('output.json', 'w') as file:
    json.dump(result, file, indent=4)
