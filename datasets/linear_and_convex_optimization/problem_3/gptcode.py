
import json
from gurobipy import *

# 1. Read Data
with open('data.json') as f:
    data = json.load(f)

# Extract data
A = range(len(data['costs']))
costs = data['costs']
young_clicks = data['young_clicks']
old_clicks = data['old_clicks']
unique_clicks = data['unique_clicks']
max_clicks = data['max_clicks']
goal_young = data['goal_young']
goal_old = data['goal_old']
goal_unique_young = data['goal_unique_young']
goal_unique_old = data['goal_unique_old']

# 2. Solver Instructions
model = Model()

# Add variables
x = model.addVars(A, name="x")

# Set objective
model.setObjective(quicksum(costs[a]*x[a] for a in A), GRB.MINIMIZE)

# 3. Add constraints
# Constraint for total clicks from young visitors
model.addConstr(quicksum(young_clicks[a]*x[a] for a in A) >= goal_young)

# Constraint for total clicks from old visitors
model.addConstr(quicksum(old_clicks[a]*x[a] for a in A) >= goal_old)

# Constraint for unique clicks from young visitors
model.addConstr(quicksum(young_clicks[a]*unique_clicks[a]*x[a] for a in A) >= goal_unique_young)

# Constraint for unique clicks from old visitors
model.addConstr(quicksum(old_clicks[a]*unique_clicks[a]*x[a] for a in A) >= goal_unique_old)

# Constraint for maximum allowable clicks for each ad type
for a in A:
    model.addConstr(x[a] <= max_clicks[a])

# Solve the model
model.optimize()

# 4. Output File
output = {
    "clicks": [x[a].X for a in A],
    "total_cost": model.ObjVal
}

# Save the results
with open("output.json", "w") as f:
    json.dump(output, f, indent=4)
