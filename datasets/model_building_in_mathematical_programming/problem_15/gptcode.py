
import json
from gurobipy import *

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

x_values = data["x"]
y_values = data["y"]
K = len(x_values)

# Create a new model
model = Model("quadratic_fit")

# Create variables
c = model.addVar(name="c")
b = model.addVar(name="b")
a = model.addVar(name="a")
d = model.addVars(K, name="d")

# Set objective
model.setObjective(sum(d[k] for k in range(K)), GRB.MINIMIZE)

# Add constraints
for k in range(K):
    model.addConstr(d[k] >= y_values[k] - (c*x_values[k]**2 + b*x_values[k] + a))
    model.addConstr(d[k] >= (c*x_values[k]**2 + b*x_values[k] + a) - y_values[k])

# Optimize model
model.optimize()

# Get the values for c, b, a
quadratic = c.x
linear = b.x
constant = a.x

# Write the output to a json file
output = {"quadratic": quadratic, "linear": linear, "constant": constant}
with open('output.json', 'w') as file:
    json.dump(output, file, indent=4)
