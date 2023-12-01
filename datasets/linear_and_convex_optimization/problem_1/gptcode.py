
import json
from gurobipy import Model, GRB

# Read data
with open('data.json', 'r') as f:
    data = json.load(f)

# Data preparation
available = data['available']
requirements = data['requirements']
prices = data['prices']
costs = data['costs']
demands = data['demands']

# Number of products and raw materials
M = len(prices)
N = len(available)

# Create a new model
model = Model('WildSports')

# Create variables
x = model.addVars(M, vtype=GRB.CONTINUOUS, name="x")

# Set objective
model.setObjective(sum((prices[j] - costs[j]) * x[j] for j in range(M)), GRB.MAXIMIZE)

# Add raw material availability constraints
for i in range(N):
    model.addConstr(sum(requirements[j][i] * x[j] for j in range(M)) <= available[i])

# Add demand constraints
for j in range(M):
    model.addConstr(x[j] <= demands[j])

# Optimize model
model.optimize()

# Prepare the results
amount = [x[j].X for j in range(M)]
total_profit = model.objVal

# Save the results
output = {"amount": amount, "total_profit": total_profit}
with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
