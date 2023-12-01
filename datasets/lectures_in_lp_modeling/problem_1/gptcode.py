
import json
from gurobipy import Model, GRB

# Read data
with open("data.json", "r") as data_file:
    data = json.load(data_file)

# Get data
produce_time = data["produce_time"]
available_time = data["available_time"]
profit = data["profit"]

# Number of products and stages
K, S = len(profit), len(available_time)

# Create a new model
model = Model("product_production")

# Create variables
x = model.addVars(K, vtype=GRB.CONTINUOUS, name="quantity")

# Set objective
model.setObjective(sum(profit[k] * x[k] for k in range(K)), GRB.MAXIMIZE)

# Add constraints for available time at each stage
for s in range(S):
    model.addConstr(sum(produce_time[k][s] * x[k] for k in range(K)) <= available_time[s], "Time_{}".format(s))

# Optimize model
model.optimize()

# Get the optimal production quantity for each product
quantity = [x[k].X for k in range(K)]

# Save the results
with open("output.json", "w") as output_file:
    json.dump({"quantity": quantity}, output_file, indent=4)
