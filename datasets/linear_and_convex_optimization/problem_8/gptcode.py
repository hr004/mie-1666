
import json
from gurobipy import Model, GRB

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

time_required = data["time_required"]
machine_costs = data["machine_costs"]
availability = data["availability"]
prices = data["prices"]
min_batches = data["min_batches"]
extra_costs = data["extra_costs"]
max_extra = data["max_extra"]

M = len(machine_costs)  # number of machines
P = len(prices)  # number of parts

# Create a new model
model = Model("Auto Parts Manufacturer")

# Add variables
batches = model.addVars(P, vtype=GRB.INTEGER, name="batches")
extra_time = model.addVars(M, vtype=GRB.INTEGER, name="extra_time")

# Set objective
model.setObjective(sum(prices[p] * batches[p] for p in range(P)) -
                   sum(machine_costs[m] * sum(time_required[m][p] * batches[p] for p in range(P)) for m in range(M)) -
                   sum(extra_costs[m] * extra_time[m] for m in range(M)), GRB.MAXIMIZE)

# Add constraints
for m in range(M):
    model.addConstr(sum(time_required[m][p] * batches[p] for p in range(P)) + extra_time[m] <= availability[m] + max_extra[m])
    model.addConstr(sum(time_required[m][p] * batches[p] for p in range(P)) <= availability[m] + extra_time[m])

for p in range(P):
    model.addConstr(batches[p] >= min_batches[p])

# Optimize model
model.optimize()

# Extract results
batches_result = [batches[p].X for p in range(P)]
extra_time_result = [extra_time[m].X for m in range(M)]
total_profit = model.objVal

# Save output
output = {
    "batches": batches_result,
    "extra_time": extra_time_result,
    "total_profit": total_profit
}

with open("output.json", "w") as file:
    json.dump(output, file, indent=4)
