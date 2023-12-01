
import json
from gurobipy import Model, GRB, quicksum

# Step 1: Read Data
with open('data.json') as f:
    data = json.load(f)

K = int(data["K"])
p = list(map(float, data["p"]))
E_Z = float(data["E[Z]"])
E_Z2 = float(data["E[Z^2]"])

# Step 2: Create a new model
model = Model()

# Step 3: Create Variables
x = model.addVars(K+1, name="x")

# Step 4: Set Objective - Lower Bound
model.setObjective(quicksum(x[k] for k in range(K+1)), GRB.MINIMIZE)

# Step 5: Add Constraints
model.addConstr(quicksum((k**4)*x[k] for k in range(K+1)) == 1)
model.addConstr(quicksum(k*x[k] for k in range(K+1)) == E_Z)
model.addConstr(quicksum((k**2)*x[k] for k in range(K+1)) == E_Z2)

# Step 6: Solve
model.optimize()

# Check if the model is optimal before calling the ObjVal attribute
if model.status == GRB.OPTIMAL:
    # Step 7: Save the lower bound
    lower_bound = model.ObjVal  # Correct attribute is 'ObjVal'

    # Step 8: Set Objective - Upper Bound
    model.setObjective(quicksum(x[k] for k in range(K+1)), GRB.MAXIMIZE)

    # Step 9: Solve
    model.optimize()

    # Step 10: Save the upper bound
    upper_bound = model.ObjVal  # Correct attribute is 'ObjVal'

    # Step 11: Write the output
    output = {"lower_bound": lower_bound, "upper_bound": upper_bound}
    with open('output.json', 'w') as f:
        json.dump(output, f, indent=4)
else:
    print('The model is not optimal. Please check the constraints and try again.')
