
import json
from gurobipy import *

# Read Data
with open("data.json", "r") as file:
    data = json.load(file)

# Extract data
time_required = data["time_required"]
machine_costs = data["machine_costs"]
availability = data["availability"]
prices = data["prices"]
min_batches = data["min_batches"]

# Create a new model
model = Model("auto_parts_manufacturing")

# Number of parts and machines
P, M = len(prices), len(machine_costs)

# Decision Variables
batches = model.addVars(P, lb=0, name="batches")

# Objective Function
model.setObjective(
    quicksum(prices[p] * batches[p] for p in range(P)) - 
    quicksum(
        machine_costs[m] * 
        quicksum(time_required[m][p] * batches[p] for p in range(P)) 
        for m in range(M)
    ),
    GRB.MAXIMIZE
)

# Constraints
for p in range(P):
    model.addConstr(batches[p] >= min_batches[p], f"MinBatches_{p}")

for m in range(M-1):
    model.addConstr(
        quicksum(time_required[m][p] * batches[p] for p in range(P)) <= availability[m], 
        f"MachineAvailability_{m+1}"
    )

model.addConstr(
    quicksum(time_required[M-2][p] * batches[p] for p in range(P)) + 
    quicksum(time_required[M-1][p] * batches[p] for p in range(P)) <= 
    availability[M-2] + availability[M-1], 
    "SharedAvailability_M_M-1"
)

# Solve Model
model.optimize()

if model.status == GRB.OPTIMAL:
    # Extract solution
    solution = {
        "batches": [batches[p].x for p in range(P)],
        "total_profit": model.objVal
    }

    # Write solution to file
    with open("output.json", "w") as file:
        json.dump(solution, file, indent=4)
