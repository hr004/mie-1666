
import json
from gurobipy import *

# 1. Read Data
with open('data.json', 'r') as f:
    data = json.load(f)

C = data['C']
value = data['value']
size = data['size']
K = len(value)

# 2. Create a new model
model = Model("knapsack")

# 3. Create variables
x = model.addVars(K, vtype=GRB.BINARY, name="x")

# 4. Set objective
model.setObjective(quicksum(value[k] * x[k] for k in range(K)), GRB.MAXIMIZE)

# 5. Add capacity constraint
model.addConstr(quicksum(size[k] * x[k] for k in range(K)) <= C, "Capacity")

# 6. Optimize model
model.optimize()

# 7. Write results to the output file
output = {"isincluded": [int(x[k].x) for k in range(K)]}
with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
