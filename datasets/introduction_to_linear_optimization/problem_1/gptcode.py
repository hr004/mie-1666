
import json
from gurobipy import Model, GRB

# Read data
with open("data.json", "r") as file:
    data = json.load(file)

available = data["available"]
requirements = data["requirements"]
prices = data["prices"]

# Number of goods and raw materials
M = len(prices)
N = len(available)

# Create a new model
model = Model("firm_model")

# Create variables
X = model.addVars(M, lb=0, vtype=GRB.CONTINUOUS, name="X")

# Set objective
model.setObjective(sum(prices[j]*X[j] for j in range(M)), GRB.MAXIMIZE)

# Add constraints for each raw material
for i in range(N):
    model.addConstr(sum(requirements[j][i]*X[j] for j in range(M)) <= available[i])

# Optimize model
model.optimize()

# Get the optimal solution
amount = [X[j].x for j in range(M)]

# Save the results
with open("output.json", "w") as file:
    json.dump({"amount": amount}, file, indent=4)
