
import gurobipy as gp
import json

# Read data
with open('data.json') as data_file:
    data = json.load(data_file)

# Extract routes, supply, and demand from data
routes = data["routes"]
supply = {item["terminal"]: item["S"] for item in data["supply"]}
demand = {item["destination"]: item["D"] for item in data["demand"]}

# Get all unique cities
cities = set(
    [route["start"] for route in routes] + [route["end"] for route in routes]
)

# Create cost dictionary
C = {(route["start"], route["end"]): route["C"] for route in routes}

# Create a model
model = gp.Model()

# add variables
x = model.addVars(
    ((route["start"], route["end"]) for route in routes),
    obj=[C[(route["start"], route["end"])] for route in routes],
    name="x"
)

# add supply constraints
model.addConstrs(
    (gp.quicksum(x[k, j] for j in cities if (k, j) in x) <= supply[k] for k in supply),
    name="supply"
)

# add demand constraints
model.addConstrs(
    (gp.quicksum(x[i, l] for i in cities if (i, l) in x) >= demand[l] for l in demand),
    name="demand"
)

# optimize the model
model.optimize()

# print the optimal solution
for v in model.getVars():
    print(f'{v.varName}, {v.x}')

print(f'Objective: {model.objVal}')
