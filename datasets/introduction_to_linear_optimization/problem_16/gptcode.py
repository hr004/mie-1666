
import json
from pulp import *

# Load the data
with open('data.json') as data_file:
    data = json.load(data_file)

model = LpProblem("Maximize Revenue", LpMaximize)

# Variables
x = [LpVariable(f'x{l}', lowBound=0) for l in range(len(data['cost']))]

# Objective
model += lpSum([data['price'][p] * sum([data['output'][l][p] * x[l] for l in range(len(data['cost']))]) for p in range(len(data['price']))]) - lpSum([data['cost'][l] * sum([data['output'][l][p] * x[l] for p in range(len(data['price']))]) for l in range(len(data['cost']))])

# Constraints
for i in range(len(data['allocated'])):
    model += lpSum([data['input'][l][i] * x[l] for l in range(len(data['cost']))]) <= data['allocated'][i]

# Solve
model.solve()

# Get the results
revenue = value(model.objective)
execute = [value(var) for var in x]

# Save the results
with open('output.json', 'w') as output_file:
    json.dump({"revenue": revenue, "execute": execute}, output_file)
