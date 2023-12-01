
import json
from gurobipy import Model, GRB, quicksum

# Read Data
with open('data.json') as json_file:
    data = json.load(json_file)

capacity = data["capacity"]
size = data["size"]
N = len(size)

# Model
model = Model()

# Decision Variables
x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
y = model.addVars(N, vtype=GRB.BINARY, name="y")

# Objective Function
model.setObjective(quicksum(y[i] for i in range(N)), GRB.MINIMIZE)

# Constraints
for j in range(N):
    model.addConstr(quicksum(x[i, j] for i in range(N)) == 1)

for i in range(N):
    for j in range(N):
        model.addConstr(x[i, j] <= y[i])

for i in range(N):
    model.addConstr(quicksum(size[j]*x[i, j] for j in range(N)) <= capacity * y[i])

# Solve Model    
model.optimize()

n_disks = int(sum(y[i].x for i in range(N)))
whichdisk = [int(sum(i*x[i, j].x for i in range(N))) for j in range(N)]

# Write output data to json file
output = {"n_disks": n_disks, "whichdisk": whichdisk}
with open('output.json', 'w') as json_file:
    json.dump(output, json_file, indent=4)
