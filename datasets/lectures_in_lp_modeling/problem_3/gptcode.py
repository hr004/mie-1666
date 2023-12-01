
import json
from gurobipy import *

# Load data
with open('data.json') as data_file:
    data = json.load(data_file)
price = data["price"]
demand = data["demand"]
nutrition = data["nutrition"]

# Create a new model
model = Model("Balanced Diet LP")

# Create variables
x = model.addVars(len(price), lb=0, name="x")

# Set objective
model.setObjective(quicksum(price[k]*x[k] for k in range(len(price))), GRB.MINIMIZE)

# Add demand constraints
for m in range(len(demand)):
    model.addConstr(quicksum(nutrition[k][m]*x[k] for k in range(len(price))) >= demand[m])

# Optimize model
model.optimize()

# Extract solution
quantity = [x[k].x for k in range(len(price))]

# Save output
with open('output.json', 'w') as output_file:
    json.dump({"quantity": quantity}, output_file, indent=4)
