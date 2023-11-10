
import json
import gurobipy as gp
from gurobipy import GRB

# Read data from file
with open("data.json", "r") as file:
    data = json.load(file)

demand = data["demand"]
max_regular_amount = data["max_regular_amount"]
cost_regular = data["cost_regular"]
cost_overtime = data["cost_overtime"]
store_cost = data["store_cost"]
N = len(demand)

# Create model
model = gp.Model()

# Create decision variables
reg_quant = model.addVars(N, name="reg_quant", lb=0)
over_quant = model.addVars(N, name="over_quant", lb=0)
store_quant = model.addVars(N, name="store_quant", lb=0)

# Set objective function
total_cost = sum(cost_regular * reg_quant[n] + cost_overtime * over_quant[n] + store_cost * store_quant[n] for n in range(N))
model.setObjective(total_cost, GRB.MINIMIZE)

# Add constraints
for n in range(N):
    # Demand constraint
    model.addConstr(reg_quant[n] + over_quant[n] == demand[n], name=f"demand_{n+1}")
    
    # Production limit constraint
    model.addConstr(reg_quant[n] <= max_regular_amount, name=f"production_limit_{n+1}")
    
    # Storage constraint
    if n == 0:
        model.addConstr(store_quant[n] == reg_quant[n] + over_quant[n] - demand[n], name=f"storage_{n+1}")
    else:
        model.addConstr(store_quant[n] == reg_quant[n] + over_quant[n] - demand[n] + store_quant[n-1], name=f"storage_{n+1}")
    
# Optimize model
model.optimize()

# Check optimization status
if model.status == GRB.OPTIMAL:
    # Get optimal solution
    reg_quant_values = [reg_quant[n].x for n in range(N)]
    over_quant_values = [over_quant[n].x for n in range(N)]
    
    # Save output to file
    output = {
        "reg_quant": reg_quant_values,
        "over_quant": over_quant_values
    }
    
    with open("output.json", "w") as file:
        json.dump(output, file, indent=4)
