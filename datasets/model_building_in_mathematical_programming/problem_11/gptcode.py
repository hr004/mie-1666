
import json
from gurobipy import *

# Read the data
with open("data.json", "r") as file:
    data = json.load(file)

# Parameters
inputone = data["inputone"]
manpowerone = data["manpowerone"]
inputtwo = data["inputtwo"]
manpowertwo = data["manpowertwo"]
stock = data["stock"]
init_capacity = data["capacity"]  # initial capacity
demand = data["demand"]

K = len(inputone)  # Number of industries
T = 5  # Number of years

# Create a new model
model = Model("economy")

# Create variables
produce = [[model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"produce_{k}_{t}") for t in range(T)] for k in range(K)]
buildcapa = [[model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"buildcapa_{k}_{t}") for t in range(T)] for k in range(K)]
stockhold = [[model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"stockhold_{k}_{t}") for t in range(T)] for k in range(K)]
capacity = [[model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"capacity_{k}_{t}") for t in range(T)] for k in range(K)]  # capacity needs to be a variable

# Set objective
model.setObjective(quicksum(manpowerone[k]*produce[k][t] + manpowertwo[k]*buildcapa[k][t] for k in range(K) for t in range(T)), GRB.MAXIMIZE)

# Add production and stock constraints
for k in range(K):
    for t in range(T):
        model.addConstr(produce[k][t] <= capacity[k][t], f"Production_Capacity_{k}_{t}")
        if t == 0:
            model.addConstr(stockhold[k][t] == stock[k] + produce[k][t] - demand[k] - buildcapa[k][t], f"Stock_Balance_{k}_{t}")
            model.addConstr(capacity[k][t] == init_capacity[k], f"Initial_Capacity_{k}")  # initial capacity constraint
        else:
            model.addConstr(stockhold[k][t] == stockhold[k][t-1] + produce[k][t] - demand[k] - buildcapa[k][t], f"Stock_Balance_{k}_{t}")
            model.addConstr(capacity[k][t] == capacity[k][t-1] + buildcapa[k][t-1], f"Capacity_{k}_{t}")  # correct capacity constraints

# Optimize model
model.optimize()

# Check if the model is optimal
if model.Status == GRB.OPTIMAL:
    # Save results
    result = {
        "produce": [[produce[k][t].X for t in range(T)] for k in range(K)],
        "buildcapa": [[buildcapa[k][t].X for t in range(T)] for k in range(K)],
        "stockhold": [[stockhold[k][t].X for t in range(T)] for k in range(K)],
        "capacity": [[capacity[k][t].X for t in range(T)] for k in range(K)]  # capacity values will also be part of the results
    }

    with open("output.json", "w") as file:
        json.dump(result, file, indent=4)
else:
    print('The model cannot be solved. The status code is: ', model.Status)
