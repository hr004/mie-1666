
import json
from gurobipy import *

# Step 1: Read Data
with open('data.json') as data_file:
    data = json.load(data_file)

coeff = data["coeff"]
desired = data["desired"]

N = len(desired)
M = len(coeff[0])

# Step 2: Create a new model
model = Model("illumination")

# Step 3: Add decision variables
power = model.addVars(M, lb=0, name="power")
error = model.addVars(N, lb=0, name="error")

# Step 4: Add objective
model.setObjective(error.sum(), GRB.MINIMIZE)

# Step 5: Add constraints
for i in range(N):
    model.addConstr(quicksum(coeff[i][j] * power[j] for j in range(M)) - error[i] <= desired[i])
    model.addConstr(quicksum(coeff[i][j] * power[j] for j in range(M)) + error[i] >= desired[i])

# Step 6: Update the model
model.update()

# Step 7: Solve the model
model.optimize()

# Step 8: Extract solution
solution = [power[j].x for j in range(M)]
total_error = model.objVal

# Step 9: Save results
output = {"power": solution, "error": total_error}
with open('output.json', 'w') as output_file:
    json.dump(output, output_file, indent=4)
