
import json
from gurobipy import *

# Read data
with open("data.json", "r") as data_file:
    data = json.load(data_file)

# Extract data
K = len(data["strength"])
I = len(data["requirement"][0])
requirement = data["requirement"]
strength = data["strength"]
lessonewaste = data["lessonewaste"]
moreonewaste = data["moreonewaste"]
recruit_limit = data["recruit"]
costredundancy = data["costredundancy"]
num_overman = data["num_overman"]
costoverman = data["costoverman"]
num_shortwork = data["num_shortwork"]
costshort = data["costshort"]

# Create model
model = Model()

# Add decision variables
recruit = model.addVars(K, I, name="recruit")
overmanning = model.addVars(K, I, name="overmanning")
short = model.addVars(K, I, name="short")

# Set objective function
model.setObjective(
    quicksum(costredundancy[k] * (strength[k] - quicksum(recruit[k, i] + overmanning[k, i] - short[k, i] for i in range(I)) for k in range(K)),
    GRB.MINIMIZE
)

# Add constraints
for k in range(K):
    for i in range(I):
        # Manpower requirement constraint
        model.addConstr(
            strength[k] * (1 - moreonewaste[k]) + quicksum((recruit[k, j] + overmanning[k, j] - short[k, j]) * (1 - lessonewaste[k]) for j in range(i+1)) >= requirement[k][i],
            name=f"manpower_req_{k}_{i}"
        )

        # Recruitment constraint
        model.addConstr(
            recruit[k, i] <= recruit_limit[k],
            name=f"recruit_limit_{k}_{i}"
        )

    # Short-time working constraint
    model.addConstr(
        quicksum(short[k, i] for i in range(I)) <= num_shortwork,
        name=f"short_limit_{k}"
    )

# Overmanning constraint
model.addConstr(
    quicksum(overmanning[k, i] for k in range(K