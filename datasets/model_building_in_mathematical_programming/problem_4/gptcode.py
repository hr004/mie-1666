
from gurobipy import *
import json

# Read data from the file
with open("data.json", "r") as read_file:
    data = json.load(read_file)

down = data['down']
m = data['maintain']  # assume this is a 2D list or a dictionary
M = len(down)
I = len(m[0])  # assuming all machines have the same number of months

total_down_time_cost = quicksum(down[n]*m[n][i] for n in range(M) for i in range(I))
