
import json
from gurobipy import *

# Read Data
with open('data.json', 'r') as f:
    data = json.load(f)

T = len(data["deliver"])
storage_cost = data["storage_cost"]
switch_cost = data["switch_cost"]
deliver = data["deliver"]

# Model
model = Model('Production_and_Inventory')

# Decision Variables
x = model.addVars(T, vtype=GRB.INTEGER, name="x")
I = model.addVars(T, vtype=GRB.INTEGER, name="I")
d = model.addVars(T, vtype=GRB.INTEGER, name="d")  # Additional variable for the absolute difference

# Objective Function
model.setObjective(storage_cost * sum(I[i] for i in range(T)) + 
                   switch_cost * sum(d[i] for i in range(1, T)), GRB.MINIMIZE)

# Constraints
model.addConstr(I[0] == 0)  # Initial inventory
I_previous = I[0]
for i in range(1, T):
    model.addConstr(x[i] - x[i-1] <= d[i])  # d[i] is greater than or equal to x[i] - x[i-1]
    model.addConstr(x[i-1] - x[i] <= d[i])  # d[i] is greater than or equal to x[i-1] - x[i]
    model.addConstr(x[i] + I_previous == deliver[i] + I[i])  # Inventory balance
    I_previous = I[i]
model.addConstr(x[0] + I_previous == deliver[0] + I[0])  # Inventory balance for the first month
model.addConstr(I[T-1] == 0)  # No remaining inventory

# Solve
model.optimize()

# Extract solution
solution = {}
solution["x"] = [int(x[i].x) for i in range(T)]
solution["cost"] = int(model.objVal)

# Save solution
with open('output.json', 'w') as f:
    json.dump(solution, f, indent=4)
