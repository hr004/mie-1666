
import json
from gurobipy import *

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

T = range(len(data['demand']))
K = range(len(data['num']))

# Create a new model
model = Model("power_station_problem")

# Create variables
x = model.addVars(K, T, vtype=GRB.BINARY, name="x")
y = model.addVars(K, T, vtype=GRB.INTEGER, name="y")

# Set objective
model.setObjective(
    quicksum(
        data['runcost'][k] * x[k, t] +
        data['extracost'][k] * (y[k, t] - data['minlevel'][k] * x[k, t]) +
        data['startcost'][k] * (x[k, t] - (x[k, t-1] if t > 0 else 0))
        for k in K for t in T
    ),
    GRB.MINIMIZE
)

# Add demand constraint
model.addConstrs(
    (quicksum(y[k, t] for k in K) >= data['demand'][t] for t in T),
    name="demand"
)

# Add generator capacity constraint
model.addConstrs(
    (data['minlevel'][k] * x[k, t] <= y[k, t] for k in K for t in T),
    name="capacity_min"
)

model.addConstrs(
    (y[k, t] <= data['maxlevel'][k] * x[k, t] for k in K for t in T),
    name="capacity_max"
)

# Add generator number constraint
model.addConstrs(
    (x[k, t] <= data['num'][k] for k in K for t in T),
    name="num"
)

# Optimize model
model.optimize()

# Check if a feasible solution is found
if model.status == GRB.Status.OPTIMAL:
    # Prepare the solution in the correct format
    solution = {'numon': [[int(x[k, t].x) for t in T] for k in K]}

    # Write the solution to the output file
    with open('output.json', 'w') as file:
        json.dump(solution, file, indent=4)
