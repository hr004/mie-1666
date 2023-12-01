
import json
from gurobipy import *

# Read Data
with open('data.json') as data_file:
    data = json.load(data_file)
officers_needed = [int(val) for val in data["officers_needed"]]
shift_costs = [int(val) for val in data["shift_costs"]]
S = len(officers_needed)

# Create a new model
model = Model("PoliceShift")

# Create Variables
x = model.addVars(S, vtype=GRB.INTEGER, name="x")

# Set objective
model.setObjective(sum(shift_costs[s]*x[s] for s in range(S)), GRB.MINIMIZE)

# Add demand constraints
model.addConstr(x[S-1] + x[0] >= officers_needed[0], "Demand0")
for s in range(1, S):
    model.addConstr(x[s-1] + x[s] >= officers_needed[s], f"Demand{s}")

# Optimize model
model.optimize()

# Get results
officers_assigned = [int(x[s].x) for s in range(S)]
total_cost = int(model.objVal)

# Save results
results = {"officers_assigned": officers_assigned, "total_cost": total_cost}
with open('output.json', 'w') as outfile:
    json.dump(results, outfile, indent=4)
