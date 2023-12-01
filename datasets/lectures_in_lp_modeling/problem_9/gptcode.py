
import json
from gurobipy import *

# Step 1: Read Data
with open('data.json') as data_file:
    data = json.load(data_file)

n_steel_quant = data["n_steel_quant"]
mn_percent = data["mn_percent"]
si_min = data["si_min"]
si_max = data["si_max"]
contsi = data["contsi"]
contmn = data["contmn"]
mang_price = data["mang_price"]
cost = data["cost"]
sell_price = data["sell_price"]
melt_price = data["melt_price"]

K = len(contsi)  # Number of minerals

# Step 2: Create the model
model = Model('Steel production')

# Step 3: Add decision variables
x = model.addVars(K, lb=0, name="x")  # Amount of each mineral melted
y = model.addVar(lb=0, name="y")  # Amount of Manganese directly added

# Step 4: Set the objective function
model.setObjective(n_steel_quant * sell_price - quicksum((x[k] * cost[k] + x[k] * melt_price)
                                                         for k in range(K)) - y * mang_price, GRB.MAXIMIZE)

# Step 5: Add constraints
# Total steel quantity constraint
model.addConstr(quicksum(x[k] for k in range(K)) + y == n_steel_quant * 10**3)

# Manganese percentage constraint
model.addConstr(quicksum(x[k] * contmn[k] for k in range(K)) + y >= mn_percent * n_steel_quant)

# Silicon percentage constraints
model.addConstr(quicksum(x[k] * contsi[k] for k in range(K)) >= si_min * n_steel_quant)
model.addConstr(quicksum(x[k] * contsi[k] for k in range(K)) <= si_max * n_steel_quant)

# Step 6: Solve the model
model.optimize()

# Step 7: Write results to the output file
output = {"amount": [x[k].x for k in range(K)], "num_mang": y.x}
with open('output.json', 'w') as output_file:
    json.dump(output, output_file, indent=4)
