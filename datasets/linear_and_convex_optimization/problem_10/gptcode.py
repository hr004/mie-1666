
import json
from gurobipy import Model, GRB

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

# Create a new model
model = Model("Electricity_Transmission")

# Create variables
x = {}
for p, supply in enumerate(data["supply"]):
    for c, demand in enumerate(data["demand"]):
        x[p, c] = model.addVar(lb=0, ub=supply, name=f"x[{p},{c}]")

# Set objective
model.setObjective(
    sum(x[p, c]*data["transmission_costs"][p][c] for p in range(len(data["supply"])) for c in range(len(data["demand"]))),
    GRB.MINIMIZE)

# Add demand constraints
for c in range(len(data["demand"])):
    model.addConstr(sum(x[p, c] for p in range(len(data["supply"]))) == data["demand"][c])

# Add supply constraints
for p in range(len(data["supply"])):
    model.addConstr(sum(x[p, c] for c in range(len(data["demand"]))) <= data["supply"][p])

# Optimize model
model.optimize()

# Prepare output data
output_data = {
    "send": [[x[p, c].x for c in range(len(data["demand"]))] for p in range(len(data["supply"]))],
    "total_cost": model.objVal
}

# Write output data to file
with open('output.json', 'w') as file:
    json.dump(output_data, file, indent=4)
