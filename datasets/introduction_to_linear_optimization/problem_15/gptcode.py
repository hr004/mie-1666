
import json
from gurobipy import GRB, Model

# Read data
with open('data.json', 'r') as file:
    data = json.load(file)

N = len(data["assemblyHour"])  # Number of products
materialDiscount = float(data["materialDiscount"]) / 100  # Convert percentage to decimal
discountThreshold = float(data["discountThreshold"])

# Create a new model
model = Model('dailyProfitOptimization')

# Create variables
X = model.addVars(N, vtype=GRB.INTEGER, name="unitsProduced")
O = model.addVar(vtype=GRB.INTEGER, name="overtimeAssembly")
M = model.addVar(name="materialBought")
Z = model.addVar(vtype=GRB.BINARY, name="discountIndicator")

# Big-M value
bigM = sum(data["materialCost"][i] for i in range(N))

# Set objective
model.setObjective(sum(data["price"][i] * X[i] for i in range(N)) - M * (1 - materialDiscount * Z) - O * float(data["overtimeAssemblyCost"]), GRB.MAXIMIZE)

# Add constraints
model.addConstr(sum(data["assemblyHour"][i] * X[i] for i in range(N)) + O <= float(data["maxAssembly"]) + float(data["maxOvertimeAssembly"]), "assemblyHours")
model.addConstr(sum(data["testingHour"][i] * X[i] for i in range(N)) <= float(data["maxTesting"]), "testingHours")
model.addConstr(sum(data["materialCost"][i] * X[i] for i in range(N)) <= M, "rawMaterials")
model.addConstr(M - bigM * Z <= discountThreshold, "discountThreshold1")
model.addConstr(M - bigM * (1 - Z) >= discountThreshold, "discountThreshold2")

# Optimize model
model.optimize()

# Prepare output
output = {
    "dailyProfit": model.objVal,
    "unitsProduced": [int(X[i].X) for i in range(N)],
    "overtimeAssembly": int(O.X),
    "materialBought": M.X
}

# Save output
with open('output.json', 'w') as file:
    json.dump(output, file, indent=4)
