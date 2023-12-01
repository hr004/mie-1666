
import json
from gurobipy import *

# Step 1: Read Data
with open('data.json') as f:
    data = json.load(f)

cash = data["cash"]
hour = data["hour"]
cost = data["cost"]
price = data["price"]
investPercentage = data["investPercentage"]
upgradeHours = data["upgradeHours"]
upgradeCost = data["upgradeCost"]
availableHours = data["availableHours"]

P = len(price) # number of products

# Step 2: Create a new model
model = Model("Company Optimization Model")

# Step 3: Create Variables
x = model.addVars(P, lb=0, vtype=GRB.CONTINUOUS, name="production") # Production quantities
y = model.addVar(vtype=GRB.BINARY, name="upgrade") # Upgrade decision

# Step 4: Set objective
model.setObjective(quicksum((price[i] - cost[i] - investPercentage[i] * price[i]) * x[i] for i in range(P)) - upgradeCost * y, GRB.MAXIMIZE)

# Step 5: Add constraints
model.addConstr(quicksum(cost[i] * x[i] + investPercentage[i] * price[i] * x[i] for i in range(P)) + upgradeCost * y <= cash, "Cash_Constraint")
model.addConstr(quicksum(hour[i] * x[i] for i in range(P)) <= availableHours + upgradeHours * y, "Hour_Constraint")

# Step 6: Optimize model
model.optimize()

# Step 7: Print and save results
if model.status == GRB.OPTIMAL:  # Ensure the model has been solved to optimality
    net_income = model.objVal
    production = [x[i].x for i in range(P)]
    upgrade = y.x

    output = {
        "net_income": net_income,
        "production": production,
        "upgrade": bool(upgrade),
    }

    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)
else:
    print("No optimal solution found.")
