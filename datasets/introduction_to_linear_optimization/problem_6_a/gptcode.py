
import json
import gurobipy as gp
from gurobipy import GRB

# Read data from json file
with open("data.json") as f:
    data = json.load(f)

# Parse data
x_0 = data['x_0']
v_0 = data['v_0']
x_T = data['x_T']
v_T = data['v_T']
T = data['T']

# Create a new model
model = gp.Model("RocketModel")

# Create variables
a_pos = model.addVars(T, lb=0, name="a_pos")
a_neg = model.addVars(T, lb=0, name="a_neg")
x = model.addVars(T + 1, lb=-GRB.INFINITY, name="x")
v = model.addVars(T + 1, lb=-GRB.INFINITY, name="v")

# Set objective
model.setObjective(gp.quicksum(a_pos[t] + a_neg[t] for t in range(T)), GRB.MINIMIZE)

# Add constraints
model.addConstrs((x[t] == x[t - 1] + v[t - 1] for t in range(1, T + 1)), "position")
model.addConstrs((v[t] == v[t - 1] + a_pos[t - 1] - a_neg[t - 1] for t in range(1, T + 1)), "velocity")
model.addConstr(x[0] == x_0, "initial_position")
model.addConstr(v[0] == v_0, "initial_velocity")
model.addConstr(x[T] == x_T, "final_position")
model.addConstr(v[T] == v_T, "final_velocity")

# Optimize model
model.optimize()

# Extract solution
solution = {
    "x": [x[t].X for t in range(T + 1)],
    "v": [v[t].X for t in range(T + 1)],
    "a": [(a_pos[t].X - a_neg[t].X) for t in range(T)],
    "fuel_spend": model.objVal,
}

# Save output to json file
with open("output.json", 'w') as f:
    json.dump(solution, f, indent=4)
