
import json
from gurobipy import Model, GRB

# Read Data
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract data lists
x = data['x']
y = data['y']

# Create a model
model = Model()

# Add decision variables
a = model.addVar(vtype=GRB.CONTINUOUS, name='a')
b = model.addVar(vtype=GRB.CONTINUOUS, name='b')
z = model.addVar(vtype=GRB.CONTINUOUS, name='z')

# Set objective
model.setObjective(z, GRB.MINIMIZE)

# Add constraints
for k in range(len(x)):
    model.addConstr(a + b * x[k] - y[k] <= z, name=f'constraint_{k}_1')
    model.addConstr(y[k] - (a + b * x[k]) <= z, name=f'constraint_{k}_2')

# Optimize
model.optimize()

# Get the optimal values
intercept = a.X
slope = b.X

# Save the results
with open('output.json', 'w') as file:
    json.dump({"intercept": intercept, "slope": slope}, file, indent=4)
