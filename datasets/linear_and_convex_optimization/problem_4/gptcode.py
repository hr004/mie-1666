
import json
from gurobipy import Model, GRB

# Read data
with open("data.json", "r") as data_file:
    data = json.load(data_file)

# Create a new model
model = Model("steel_production")

# Get data
A = len(data["available"])
S = len(data["carbon_min"])

# Decision variables
x = [[model.addVar(lb=0.0, vtype=GRB.CONTINUOUS) for _ in range(A)] for _ in range(S)]
y = [model.addVar(lb=0.0, vtype=GRB.CONTINUOUS) for _ in range(S)]

# Set objective
model.setObjective(sum(y[s]*float(data["steel_prices"][s]) for s in range(S)) - 
                   sum(x[s][a]*float(data["alloy_prices"][a]) for s in range(S) for a in range(A)), GRB.MAXIMIZE)

# Add constraints
for a in range(A):
    model.addConstr(sum(x[s][a] for s in range(S)) <= float(data["available"][a]))

for s in range(S):
    model.addConstr(sum(x[s][a]*float(data["carbon"][a]) for a in range(A)) >= y[s]*float(data["carbon_min"][s]))
    model.addConstr(sum(x[s][a]*float(data["nickel"][a]) for a in range(A)) <= y[s]*float(data["nickel_max"][s]))
    model.addConstr(sum(x[s][a] for a in range(A)) == y[s])
    model.addConstr(x[s][0] <= 0.4*y[s])

# Optimize model
model.optimize()

# Prepare output data
output = {
    "alloy_use": [[x[s][a].x for a in range(A)] for s in range(S)],
    "total_steel": [y[s].x for s in range(S)],
    "total_profit": model.objVal
}

# Save output data
with open("output.json", "w") as output_file:
    json.dump(output, output_file, indent=4)
