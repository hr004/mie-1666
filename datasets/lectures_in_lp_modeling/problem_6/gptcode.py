
import json
from gurobipy import Model, GRB

# Read data
with open('data.json', 'r') as f:
    data = json.load(f)

time = data['time']
profit = data['profit']
capacity = data['capacity']

# Number of parts and shops
K = len(profit)
S = len(capacity)

# Create a new model
model = Model('SparePartsProduction')

# Add variables
x = model.addVars(K, vtype=GRB.CONTINUOUS, name="x")

# Set the objective function
model.setObjective(sum(profit[k]*x[k] for k in range(K)), GRB.MAXIMIZE)

# Add capacity constraints
for s in range(S):
    model.addConstr(sum(time[k][s]*x[k] for k in range(K)) <= capacity[s], f'Capacity_{s + 1}')

# Solve the model
model.optimize()

# Check if a feasible solution is found
if model.status == GRB.OPTIMAL:
    # Get optimal solution
    quantity = [x[k].X for k in range(K)]
else:
    print("No feasible solution found")

# Save output
output = {"quantity": quantity}

with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
