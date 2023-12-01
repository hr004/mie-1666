
import json
from gurobipy import *

# Load the data
with open('data.json', 'r') as f:
    data = json.load(f)

x_values = data["x"]
y_values = data["y"]
K = len(x_values)

# Create a new model
model = Model("LinearFit")

# Create variables
a = model.addVar(name="a")
b = model.addVar(name="b")
d = model.addVars(K, name="d")

# Set objective
model.setObjective(sum(d[k] for k in range(K)), GRB.MINIMIZE)

# Add constraints
for k in range(K):
    model.addConstr(d[k] - y_values[k] + a + b*x_values[k] >= 0, "Constraint1_{}".format(k+1))
    model.addConstr(d[k] + y_values[k] - a - b*x_values[k] >= 0, "Constraint2_{}".format(k+1))

# Optimize model
model.optimize()

# Get the optimal solution
intercept = a.X
slope = b.X

# Save the output
output = {
    "intercept": intercept,
    "slope": slope
}
with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
