
import json
from gurobipy import *

# Read Data
with open('data.json') as f:
    data = json.load(f)

benefit = data["benefit"]
communication = data["communication"]
cost = data["cost"]

K = len(benefit)
L = len(benefit[0])

# Create a new model
model = Model("relocation")

# Add Variables
islocated = model.addVars(K, L, vtype=GRB.BINARY, name="islocated")

# Set Objective
model.setObjective(
    quicksum(benefit[k][l] * islocated[k, l] for k in range(K) for l in range(L)) - 
    quicksum(communication[k][j] * cost[l][m] * islocated[j, m] for k in range(K) for l in range(L) for j in range(K) for m in range(L)), 
    GRB.MINIMIZE
)

# Add Constraints
# Each department must be located in exactly one city
model.addConstrs((islocated.sum(k, '*') == 1 for k in range(K)), name="dept")

# None of the cities may be the location for more than three of the departments
model.addConstrs((islocated.sum('*', l) <= 3 for l in range(L)), name="city")

# Optimize model
model.optimize()

# Output Results
islocated_val = [[int(islocated[k, l].x) for l in range(L)] for k in range(K)]
output = {"islocated": islocated_val}

with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
