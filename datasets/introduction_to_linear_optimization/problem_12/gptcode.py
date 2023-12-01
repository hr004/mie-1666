
import json
from gurobipy import Model, GRB, quicksum
import numpy as np

# Step 1: Read Data
with open("data.json", "r") as read_file:
    data = json.load(read_file)

A = np.array(data["A"])
b = np.array(data["b"])
m, N = A.shape

# Step 2: Model Setup
model = Model("ChebychevCenter")

# Step 3: Add Decision Variables
y = model.addVars(N, lb=-GRB.INFINITY, name="Center")
r = model.addVar(name="Radius")

# Step 4: Set Objective Function
model.setObjective(r, GRB.MAXIMIZE)

# Step 5: Add Constraints
for i in range(m):
    ai_norm = np.linalg.norm(A[i])
    model.addConstr(quicksum(A[i, j] * y[j] for j in range(N)) + r * ai_norm <= b[i])

# Step 6: Solve the Model
model.optimize()

# Step 7: Retrieve Results
center = [y[j].X for j in range(N)]
radius = r.X

# Step 8: Write Output Data
output = {"center": center, "radius": radius}
with open("output.json", "w") as write_file:
    json.dump(output, write_file, indent=4)
