
import json
from gurobipy import *

# Read Data
with open('data.json', 'r') as f:
    data = json.load(f)

# Define model
model = Model('DEC')

# Define variables
N = len(data["price"])
x = model.addVars(N, lb=0, name="x")
m = model.addVars(N, lb=0, name="m")
a = model.addVars(N, lb=0, name="a")
d = model.addVars(N, lb=0, name="d")

# Set objective
model.setObjective(quicksum(data["price"][i]*x[i] for i in range(N)), GRB.MAXIMIZE)

# Constraints
for i in range(N):
    model.addConstr(x[i] <= data["demand"][i], name=f"demand_{i}")
    model.addConstr(m[i] + a[i] == data["256K_boards"][i]*x[i], name=f"mem_{i}")
    model.addConstr(d[i] == data["disk_drives"][i]*x[i], name=f"disk_{i}")
    model.addConstr(x[i] >= data["preorder"][i], name=f"preorder_{i}")

model.addConstr(quicksum(x[i] for i in range(N)) <= data["max_cpu"], name="cpu")
model.addConstr(quicksum(m[i] for i in range(N)) >= data["min_mem"], name="min_mem")
model.addConstr(quicksum(m[i] for i in range(N)) <= data["max_mem"], name="max_mem")
model.addConstr(quicksum(a[i] for i in range(N) if data["alt_compatible"][i]) <= data["alt_mem"], name="alt_mem")
model.addConstr(quicksum(d[i] for i in range(N)) >= data["min_disk"], name="min_disk")
model.addConstr(quicksum(d[i] for i in range(N)) <= data["max_disk"], name="max_disk")
model.addConstr(quicksum(x[i] for i in range(N) if not data["is_workstation"][i]) <= data["demand_GP"], name="GP")
model.addConstr(quicksum(x[i] for i in range(N) if data["is_workstation"][i]) <= data["demand_WS"], name="WS")

# Solve
model.optimize()

# Collect results
output = {
    "system_output": [
        {
            "num_produced": x[i].x,
            "total_256K_boards_used": m[i].x,
            "total_alt_boards_used": a[i].x if data["alt_compatible"][i] else 0,
            "total_disk_drives_used": d[i].x
        }
        for i in range(N)
    ],
    "profit": model.objVal
}

# Write output to file
with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
