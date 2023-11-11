
import json
from gurobipy import *

# Read data from file
with open('data.json') as f:
    data = json.load(f)

demand = data['demand']
max_regular_amount = data['max_regular_amount']
cost_regular = data['cost_regular']
cost_overtime = data['cost_overtime']
store_cost = data['store_cost']

# Create model
model = Model()

# Create decision variables
N = len(demand)
reg_quant = model.addVars(N, name="reg_quant")
over_quant = model.addVars(N, name="over_quant")

# Set objective function
total_cost = quicksum(reg_quant[n] * cost_regular + over_quant[n] * cost_overtime + demand[n] * store_cost for n in range(N))
model.setObjective(total_cost, GRB.MINIMIZE)

# Add constraints
for n in range(N):
    # Demand constraint
    model.addConstr(reg_quant[n] + over_quant[n] >= demand[n])
    
    # Regular production limit constraint
    model.addConstr(reg_quant[n] <= max_regular_amount)
    
    # Non-negativity constraint
    model.addConstr(reg_quant[n] >= 0)
    model.addConstr(over_quant[n] >= 0)

# Solve the model
model.optimize()

# Extract the solution
reg_quant_solution = [reg_quant[n].x for n in range(N)]
over_quant_solution = [over_quant[n].x for n in range(N)]

# Save the results in output.json
output = {
    "reg_quant": reg_quant_solution,
    "over_quant": over_quant_solution
}

with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
