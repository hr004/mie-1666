
import json
from gurobipy import *

# Read Data
with open('data.json', 'r') as file:
    data = json.load(file)

links = data["links"]
data_flows = data["data"]

nodes = set([link['start'] for link in links] + [link['end'] for link in links])

# Create a new model
model = Model('Network Flow Optimization')

# Decision Variables
X = model.addVars([(i, j, k, l) for i in nodes for j in nodes for k in nodes for l in nodes], 
                  vtype=GRB.CONTINUOUS, name="X")

# Objective Function
model.setObjective(
    quicksum(C['C'] * X[i, j, k, l] for C in links for i in nodes for j in nodes for k in nodes for l in nodes 
             if (C["start"] == i and C["end"] == j)), 
    GRB.MINIMIZE)

# Constraints

# Flow Conservation Constraints
for k in nodes:
    for l in nodes:
        for i in nodes:
            model.addConstr(
                quicksum(X[i, j, k, l] for j in nodes) - quicksum(X[j, i, k, l] for j in nodes) == 
                (sum(B['rate'] for B in data_flows if (B['source'] == k and B['destination'] == l)) if i == k else 
                (-sum(B['rate'] for B in data_flows if (B['source'] == k and B['destination'] == l)) if i == l else 0),
            )

# Capacity Constraints
for C in links:
    model.addConstr(
        quicksum(X[C['start'], C['end'], k, l] for k in nodes for l in nodes) <= C['U']
    )

# Non-negativity Constraints are by default in Gurobi

# Optimize model
model.optimize()

# Prepare output
optimized_paths = []

for i in nodes:
    for j in nodes:
        for k in nodes:
            for l in nodes:
                if X[i, j, k, l].X > 0:
                    optimized_paths.append({
                        "source": k,
                        "destination": l,
                        "route": [k, i, j, l],
                        "path_flow": X[i, j, k, l].X,
                        "path_cost": X[i, j, k, l].X * sum(C['C'] for C in links if (C["start"] == i and C["end"] == j))
                    })

output = {
    "optimized_paths": {
        "paths": optimized_paths,
    },
    "total_cost": model.objVal
}

# Write output
with open('output.json', 'w') as outfile:
    json.dump(output, outfile, indent=4)
