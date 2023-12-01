
import json
from gurobipy import *

# Read Data
with open("data.json", 'r') as f:
    data = json.load(f)

num = data["num"]
n_working_days = data["n_working_days"]
n_resting_days = data["n_resting_days"]
N = len(num)

# Create Model
model = Model("cafeteria")

# Add Decision Variables
x = model.addVars(N, N, vtype=GRB.BINARY, name="x")
y = model.addVars(N, N, N, vtype=GRB.BINARY, name="y")

# Set Objective
model.setObjective(sum(x[n,k] for n in range(N) for k in range(N)), GRB.MINIMIZE)

# Add Staff Requirement Constraints
for n in range(N):
    model.addConstr(sum(x[n,k] for k in range(N)) >= num[n])

# Add Work Rest Cycle Constraints
for n in range(N):
    for k in range(N):
        model.addConstr(x[n,k] == sum(y[n,k,l] for l in range(max(0, n-n_working_days-n_resting_days+1), n+1)))

# Add Work Cycle Initiation Constraints
for n in range(N):
    for k in range(N):
        model.addConstr(sum(y[n,k,l] for l in range(max(0, n-n_working_days-n_resting_days+1), n+1)) <= 1)

# Optimize Model
model.optimize()

# Check Optimization Status
if model.status == GRB.OPTIMAL:
    # Get Results
    total_number = int(model.ObjVal)
    is_work = [[int(x[n,k].x) for n in range(N)] for k in range(N)]

    # Save Results
    with open("output.json", 'w') as f:
        json.dump({"total_number": total_number, "is_work": is_work}, f, indent=4)
else:
    print('The model cannot be solved to optimality.')
