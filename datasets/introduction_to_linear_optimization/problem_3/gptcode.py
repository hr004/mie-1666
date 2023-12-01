
# Required Libraries
import json
from gurobipy import Model, GRB, quicksum

# Read Data
with open("data.json", "r") as f:
    data = json.load(f)

T = len(data["demand"])  # Number of years
demand = data["demand"]
oil_cap = data["oil_cap"]
coal_cost = data["coal_cost"]
nuke_cost = data["nuke_cost"]
max_nuke = data["max_nuke"]
coal_life = data["coal_life"]
nuke_life = data["nuke_life"]

# Initialize Model
model = Model()

# Decision Variables
coal = model.addVars(T, lb=0, name="coal")
nuke = model.addVars(T, lb=0, name="nuke")

# Objective Function
model.setObjective(quicksum(coal_cost*coal[t] + nuke_cost*nuke[t] for t in range(T)), GRB.MINIMIZE)

# Constraint: Demand
for t in range(T):
    model.addConstr(oil_cap[t] + quicksum(coal[t-i] if t-i >= 0 else 0 for i in range(min(t+1, coal_life))) +
                    quicksum(nuke[t-i] if t-i >= 0 else 0 for i in range(min(t+1, nuke_life))) >= demand[t])

# Constraint: Nuclear power capacity
for t in range(T):
    model.addConstr(quicksum(nuke[t-i] if t-i >= 0 else 0 for i in range(min(t+1, nuke_life))) <= 
                    max_nuke/100 * (oil_cap[t] + quicksum(coal[t-i] if t-i >= 0 else 0 for i in range(min(t+1, coal_life))) +
                                    quicksum(nuke[t-i] if t-i >= 0 else 0 for i in range(min(t+1, nuke_life)))))

# Optimize Model
model.optimize()

# Prepare Output
output = {
    "coal_cap_added": [coal[t].X for t in range(T)],
    "nuke_cap_added": [nuke[t].X for t in range(T)],
    "total_cost": model.objVal
}

# Write Output
with open("output.json", "w") as f:
    json.dump(output, f, indent=4)
