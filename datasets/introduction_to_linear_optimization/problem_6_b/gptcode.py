
import json
from gurobipy import *

# Read data
with open('data.json') as f:
    data = json.load(f)

x_0 = data["x_0"]
v_0 = data["v_0"]
x_T = data["x_T"]
v_T = data["v_T"]
T = data["T"]

# Create a new model
model = Model("rocket")

# Create variables
x = model.addVars(T+1, lb=-GRB.INFINITY, name="x")
v = model.addVars(T+1, lb=-GRB.INFINITY, name="v")
a = model.addVars(T, lb=-GRB.INFINITY, name="a")
f = model.addVars(T, name="f")
f_max = model.addVar(name="f_max")

# Set objective
model.setObjective(f_max, GRB.MINIMIZE)

# Add position and velocity constraints
for t in range(T):
    model.addConstr(x[t+1] == x[t] + v[t], "position_constraint[%d]" % t)
    model.addConstr(v[t+1] == v[t] + a[t], "velocity_constraint[%d]" % t)

# Add initial and final conditions constraints
model.addConstr(x[0] == x_0, "initial_position_constraint")
model.addConstr(v[0] == v_0, "initial_velocity_constraint")
model.addConstr(x[T] == x_T, "final_position_constraint")
model.addConstr(v[T] == v_T, "final_velocity_constraint")

# Add fuel consumption constraints
for t in range(T):
    model.addConstr(f[t] >= a[t], "fuel_constraint_upper[%d]" % t)
    model.addConstr(f[t] >= -a[t], "fuel_constraint_lower[%d]" % t)
    model.addConstr(f[t] <= f_max, "max_fuel_constraint[%d]" % t)

# Optimize model
model.optimize()

# Check if the model has a solution
if model.status == GRB.Status.OPTIMAL:
    # Save the results
    result = {
        "x": [x[t].X for t in range(T+1)],
        "v": [v[t].X for t in range(T+1)],
        "a": [a[t].X for t in range(T)],
        "fuel_spend": sum(f[t].X for t in range(T)),
    }

    with open('output.json', 'w') as f:
        json.dump(result, f, indent=4)
else:
    print('No solution')
