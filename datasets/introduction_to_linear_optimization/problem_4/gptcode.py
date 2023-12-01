
import gurobipy as gp
import json

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Define variables
period = data['period']
demand = data['demand']

# Create a new model
model = gp.Model()

# Define decision variables
x = model.addVars(7, lb=0, vtype=gp.GRB.INTEGER, name='x')

# Add constraints to the model
for j in range(7):
    model.addConstr(sum(x[(k%7)] for k in range(max(0, j - period + 1), j + 1)) >= demand[j])

# Define objective function
model.setObjective(sum(x[j] for j in range(7)), gp.GRB.MINIMIZE)

# Solve the model
model.optimize()

# Save the output
output = {
    'start': [x[j].x for j in range(7)],
    'total': model.objVal
}

with open('output.json', 'w') as f:
    json.dump(output, f)
