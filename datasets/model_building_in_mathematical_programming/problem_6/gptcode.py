
import json
from gurobipy import Model, GRB, quicksum

# Step 1: Read Data
with open('data.json') as data_file:
    data = json.load(data_file)

# Parameters
K = len(data["strength"])  # Number of manpower types
I = len(data["requirement"][0])  # Number of years
strength = data["strength"]
requirement = data["requirement"]
lessonewaste = data["lessonewaste"]
moreonewaste = data["moreonewaste"]
recruit_limit = data["recruit"]
costredundancy = data["costredundancy"]
num_overman = data["num_overman"]
costoverman = data["costoverman"]
num_shortwork = data["num_shortwork"]
costshort = data["costshort"]

# Step 2: Create a model
model = Model("manpower_planning")

# Step 3: Create variables
recruit = [[model.addVar(vtype=GRB.INTEGER) for _ in range(I)] for _ in range(K)]
overmanning = [[model.addVar(vtype=GRB.INTEGER) for _ in range(I)] for _ in range(K)]
short = [[model.addVar(vtype=GRB.INTEGER) for _ in range(I)] for _ in range(K)]
redundancy = [[model.addVar(vtype=GRB.INTEGER) for _ in range(I)] for _ in range(K)]

# Step 4: Set objective function
model.setObjective(quicksum(costredundancy[k]*redundancy[k][i] + costoverman[k]*overmanning[k][i] + costshort[k]*short[k][i]
                            for k in range(K) for i in range(I)), GRB.MINIMIZE)

# Step 5: Add constraints
# Constraint 1: Manpower strength
for k in range(K):
    for i in range(I):
        if i > 0:
            strength[k] = strength[k] + recruit[k][i-1] - redundancy[k][i-1]
        model.addConstr(strength[k] + recruit[k][i] - (lessonewaste[k]*recruit[k][i] + moreonewaste[k]*(strength[k] - recruit[k][i])) == overmanning[k][i] + short[k][i] + redundancy[k][i] + requirement[k][i])

# Constraint 2: Recruitment limit
for k in range(K):
    for i in range(I):
        model.addConstr(recruit[k][i] <= recruit_limit[k])

# Constraint 3: Overmanning limit
for i in range(I):
    model.addConstr(quicksum(overmanning[k][i] for k in range(K)) <= num_overman)

# Constraint 4: Short-time working limit
for k in range(K):
    for i in range(I):
        model.addConstr(short[k][i] <= num_shortwork)

# Step 6: Solve the model
model.optimize()

# Check if the model was solved successfully.
if model.status == GRB.Status.OPTIMAL:
    # Step 7: Print the solution
    output = {
        "recruit": [[recruit[k][i].X for i in range(I)] for k in range(K)],
        "overmanning": [[overmanning[k][i].X for i in range(I)] for k in range(K)],
        "short": [[short[k][i].X for i in range(I)] for k in range(K)],
    }

    # Write output to file
    with open('output.json', 'w') as output_file:
        json.dump(output, output_file, indent=4)
else:
    print('The model cannot be solved. The status code is %d. Refer to the Gurobi documentation for details.' % model.status)
