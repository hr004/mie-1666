
import json
from gurobipy import Model, quicksum, GRB

# Read Data
with open('data.json') as f:
    data = json.load(f)

# Model
model = Model('FoodManufacturing')

I = range(len(data['buy_price'][0]))  # oil types
M = range(len(data['buy_price']))  # months

# Decision Variables
B = model.addVars(I, M, name="Buy")  # Buying quantity of each oil in each month
R = model.addVars(I, M, name="Refine")  # Refined quantity of each oil in each month
S = model.addVars(I, M, name="Store")  # Storage of each oil in each month

# Objective Function
model.setObjective(quicksum(data['sell_price'] * R[i, m] for i in I for m in M) -
                   quicksum(data['buy_price'][m][i] * B[i, m] for i in I for m in M) -
                   quicksum(data['storage_cost'] * S[i, m] for i in I for m in M),
                   GRB.MAXIMIZE)

# Constraints
# Storage capacity constraints
for i in I:
    for m in M:
        model.addConstr(S[i, m] <= data['storage_size'])

# Balance constraints
for i in I:
    for m in M:
        model.addConstr(S[i, m-1] + B[i, m] == R[i, m] + S[i, m] if m > 0 else data['init_amount'] + B[i, m] == R[i, m] + S[i, m])

# Refining capacity constraints
for m in M:
    model.addConstr(quicksum(R[i, m] for i in I if data['is_vegetable'][i]) <= data['max_vegetable_refining_per_month'])
    model.addConstr(quicksum(R[i, m] for i in I if not data['is_vegetable'][i]) <= data['max_non_vegetable_refining_per_month'])

# Hardness constraints
for m in M:
    model.addConstr(data['min_hardness'] * quicksum(R[i, m] for i in I) <= quicksum(data['hardness'][i] * R[i, m] for i in I))
    model.addConstr(quicksum(data['hardness'][i] * R[i, m] for i in I) <= data['max_hardness'] * quicksum(R[i, m] for i in I))

# Final storage constraint
for i in I:
    model.addConstr(S[i, len(M)-1] == data['init_amount'])

# Solve the model
model.optimize()

# Save results
output = {"buy": [[B[i, m].x for i in I] for m in M],
          "refine": [[R[i, m].x for i in I] for m in M],
          "storage": [[S[i, m].x for i in I] for m in M]}

with open('output.json', 'w') as f:
    json.dump(output, f, indent=4)
