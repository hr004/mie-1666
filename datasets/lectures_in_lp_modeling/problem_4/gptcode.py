
import json
from gurobipy import *

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

N = len(data["cost"])
capacity = data["capacity"]
holding_cost = data["holding_cost"]
price = data["price"]
cost = data["cost"]

# Create a new model
model = Model("warehouse")

# Create variables
buyquantity = model.addVars(N, lb=0, name="buyquantity")
sellquantity = model.addVars(N, lb=0, name="sellquantity")
stock = model.addVars(N, lb=0, ub=capacity, name="stock")

# Set objective
model.setObjective(quicksum(price[n]*sellquantity[n] - cost[n]*buyquantity[n] - holding_cost*stock[n] for n in range(N)), GRB.MAXIMIZE)

# Add constraints
for n in range(1, N):
    model.addConstr(stock[n] == stock[n-1] + buyquantity[n] - sellquantity[n])

# Initial and final conditions
model.addConstr(stock[0] == buyquantity[0] - sellquantity[0])
model.addConstr(stock[N-1] == 0)

# Optimize model
model.optimize()

# If model is feasible, write solution to json file
if model.status == GRB.OPTIMAL:
    output = {
        "buyquantity": [buyquantity[n].X for n in range(N)],
        "sellquantity": [sellquantity[n].X for n in range(N)],
        "stock": [stock[n].X for n in range(N)]
    }

    with open('output.json', 'w') as file:
        json.dump(output, file, indent=4)
