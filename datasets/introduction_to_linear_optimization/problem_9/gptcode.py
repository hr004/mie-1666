
import json
from gurobipy import *

# Read data
with open('data.json') as data_file:
    data = json.load(data_file)

N = len(data["population"])
S = len(data["capacity"])
G = len(data["capacity"][0])

# Create a new model
model = Model("school_assignment")

# Add variables
x = model.addVars(N, S, G, name="x")

# Set objective
model.setObjective(quicksum(quicksum(quicksum(x[n, s, g]*data["distance"][n][s] for g in range(G)) for s in range(S)) for n in range(N)), GRB.MINIMIZE)

# Add capacity constraints
for s in range(S):
    for g in range(G):
        model.addConstr(quicksum(x[n, s, g] for n in range(N)) <= data["capacity"][s][g], f"capacity_{s}_{g}")

# Add population constraints
for n in range(N):
    for g in range(G):
        model.addConstr(quicksum(x[n, s, g] for s in range(S)) == data["population"][n][g], f"population_{n}_{g}")

# Optimize model
model.optimize()

# Save results
results = {
    "assignment": [[[x[n, s, g].X for g in range(G)] for s in range(S)] for n in range(N)],
    "total_distance": model.objVal
}

with open('output.json', 'w') as output_file:
    json.dump(results, output_file, indent=4)
