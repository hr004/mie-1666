
import json
from gurobipy import Model, GRB

# Read data
with open("data.json", "r") as data_file:
    data = json.load(data_file)

alloy_quant = data["alloy_quant"]
target = data["target"]
ratio = data["ratio"]
price = data["price"]

K = len(price)  # Number of alloys
M = len(target)  # Number of metals

# Create a new model
model = Model("Alloy Optimization")

# Create variables
x = model.addVars(K, lb=0, vtype=GRB.CONTINUOUS, name="x")

# Set objective
model.setObjective(sum(price[k] * x[k] for k in range(K)), GRB.MINIMIZE)

# Add alloy quantity constraint
model.addConstr(sum(x[k] for k in range(K)) >= alloy_quant, "AlloyQuantity")

# Add metal target constraints
for m in range(M):
    model.addConstr(sum(ratio[k][m] * x[k] for k in range(K)) >= target[m], f"Metal_{m+1}")

# Optimize model
model.optimize()

# Get the optimal solution
amount = [x[k].x for k in range(K)]

# Save the output
output = {"amount": amount}

with open("output.json", "w") as output_file:
    json.dump(output, output_file, indent=4)
