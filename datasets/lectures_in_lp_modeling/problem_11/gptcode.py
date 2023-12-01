
import json
from gurobipy import *

# Load the data
with open('data.json', 'r') as f:
    data = json.load(f)

# Create a new model
model = Model("TransportationProblem")

# Get the data
numdepot = data["numdepot"]
numport = data["numport"]
price = data["price"]
distance = data["distance"]

I = len(numdepot)
J = len(numport)

# Create variables
x = {}
for i in range(I):
    for j in range(J):
        x[i, j] = model.addVar(vtype=GRB.INTEGER, name=f"x{i}_{j}")

# Set objective
model.setObjective(quicksum(0.5 * x[i, j] * price * distance[i][j] for i in range(I) for j in range(J)), GRB.MINIMIZE)

# Add supply constraints
for i in range(I):
    model.addConstr(quicksum(x[i, j] for j in range(J)) <= numdepot[i])

# Add demand constraints
for j in range(J):
    model.addConstr(quicksum(x[i, j] for i in range(I)) >= numport[j])

# Optimize model
model.optimize()

# Check if a feasible solution is found
if model.status == GRB.OPTIMAL:
    solution = [[int(x[i, j].x) for j in range(J)] for i in range(I)]
    with open('output.json', 'w') as f:
        json.dump({"number": solution}, f, indent=4)
