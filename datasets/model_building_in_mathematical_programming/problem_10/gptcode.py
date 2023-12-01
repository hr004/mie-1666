
import json
from gurobipy import *

# Read data
with open('data.json') as f:
    data = json.load(f)

# Define parameters
K = len(data['stock'])
T = 10  # Assuming 10 years

# Create a new model
model = Model("Economy")

# Create variables
produce = model.addVars(K, T, vtype=GRB.INTEGER, name="produce")
buildcapa = model.addVars(K, T, vtype=GRB.INTEGER, name="buildcapa")
stockhold = model.addVars(K, T, vtype=GRB.INTEGER, name="stockhold")

# Set objective
model.setObjective(quicksum(produce[k, T-2] + produce[k, T-1] for k in range(K)), GRB.MAXIMIZE)

# Add constraints
for k in range(K):
    for t in range(1, T):
        model.addConstr(quicksum(data['inputone'][k][j]*produce[j, t-1] for j in range(K)) <= stockhold[k, t-1])
        model.addConstr(data['manpowerone'][k]*produce[k, t] <= data['manpower_limit'])
        if t > 1:
            model.addConstr(quicksum(data['inputtwo'][k][j]*buildcapa[j, t-2] for j in range(K)) <= stockhold[k, t-2])
        model.addConstr(data['manpowertwo'][k]*buildcapa[k, t] <= data['manpower_limit'])
        model.addConstr(produce[k, t] + buildcapa[k, t] <= stockhold[k, t])
        if t > 1:
            model.addConstr(stockhold[k, t] == stockhold[k, t-1] + produce[k, t] - quicksum(data['inputone'][j][k]*produce[j, t] + data['inputtwo'][j][k]*buildcapa[j, t] for j in range(K)))
    model.addConstr(stockhold[k, 0] == data['stock'][k])
    model.addConstr(produce[k, 0] + buildcapa[k, 0] <= data['capacity'][k])
    for t in range(2, T):
        model.addConstr(stockhold[k, t] == stockhold[k, t-1] + buildcapa[k, t-2])

# Solve model
model.optimize()

# Save the results
results = {
    "produce": [[int(produce[k, t].X) for t in range(T)] for k in range(K)],
    "buildcapa": [[int(buildcapa[k, t].X) for t in range(T)] for k in range(K)],
    "stockhold": [[int(stockhold[k, t].X) for t in range(T)] for k in range(K)]
}

with open('output.json', 'w') as f:
    json.dump(results, f, indent=4)
