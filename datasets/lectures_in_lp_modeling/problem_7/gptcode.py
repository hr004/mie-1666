
import json
import gurobipy as gp
from gurobipy import GRB

# Read data from file
with open("data.json", "r") as file:
    data = json.load(file)

# Extract data
demand = data["demand"]
max_regular_amount = data["max_regular_amount"]
cost_regular = data["cost_regular"]
cost_overtime = data["cost_overtime"]
store_cost = data["store_cost"]
N = len(demand)

# Create model
model = gp.Model()

# Create decision variables
reg_quant = model.addVars(N, lb=0, vtype=GRB.INTEGER, name="reg_quant")
over_quant = model.addVars(N, lb=0, vtype=GRB.INTEGER, name="over_quant")
store_quant = model.addVars(N-1, lb=0, vtype=GRB.INTEGER, name="store_quant")

# Set objective function
total_cost = sum(reg_quant[n] * cost_regular + over_quant[n] * cost_overtime + store_quant[n] * store_cost for n in range(N))
model.setObjective(total_cost, GRB.MINIMIZE)

# Demand constraint
demand_constraint = model.addConstrs(reg_quant[n] + over_quant[n] >= demand[n] for n in range(N))

# Maximum regular production constraint
max_regular_constraint = model.addConstrs(reg_quant[n] <= max_regular_amount for n in range(N))

# Storage constraint
storage_constraint = model.addConstrs(store_quant[n] == reg_quant[n] + over_quant[n] - demand[n] for n in range(N-1))

# Optimize model
model.optimize()

# Check optimization status
if model.status == GRB.OPTIMAL:
    # Extract solution
    reg_quant_sol = [reg_quant[n].x for n in range(N)]
    over_quant_sol = [over_quant[n].x for n in range(N)]

    # Save output to file
    output = {"reg_quant": reg_quant_sol, "over_quant": over_quant_sol}
    with open("output.json", "w") as file:
        json.dump(output, file, indent=4)
else:
    print("Optimization failed. Status:", model.status)
