
from gurobipy import *
import json

# Read data
with open("data.json", "r") as data_file:
    data = json.load(data_file)

# Create a new model
model = Model("MIP Model")

# Create variables
x = {}
y = {}
for k in range(data["n_mines"]):
    for i in range(len(data["requiredquality"])):
        x[k, i] = model.addVar(vtype=GRB.BINARY, name="x_" + str(k) + "_" + str(i))
        y[k, i] = model.addVar(vtype=GRB.CONTINUOUS, name="y_" + str(k) + "_" + str(i))

# Set the objective function
model.setObjective(
    quicksum(
        (
            data["price"] * y[k, i] - data["royalty"][k] * x[k, i]
        ) / ((1 + data["discount"]) ** i)
        for k in range(data["n_mines"])
        for i in range(len(data["requiredquality"]))
    ),
    GRB.MAXIMIZE,
)

# Add operation constraint
for i in range(len(data["requiredquality"])):
    model.addConstr(quicksum(x[k, i] for k in range(data["n_mines"])) <= data["n_maxwork"])

# Add ore limit constraint
for k in range(data["n_mines"]):
    for i in range(len(data["requiredquality"])):
        model.addConstr(y[k, i] <= data["limit"][k] * x[k, i])

# Add quality constraint
for i in range(len(data["requiredquality"])):
    model.addConstr(
        quicksum(data["quality"][k] * y[k, i] for k in range(data["n_mines"]))
        == data["requiredquality"][i]
        * quicksum(y[k, i] for k in range(data["n_mines"]))
    )

# Optimize the model
model.optimize()

# Prepare the output
output = {
    "isoperated": [
        [int(x[k, i].X) for i in range(len(data["requiredquality"]))]
        for k in range(data["n_mines"])
    ],
    "amount": [
        [y[k, i].X for i in range(len(data["requiredquality"]))]
        for k in range(data["n_mines"])
    ],
}

# Save the output
with open("output.json", "w") as output_file:
    json.dump(output, output_file, indent=4)
