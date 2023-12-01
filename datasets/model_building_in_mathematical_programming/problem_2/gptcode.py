
from gurobipy import Model, quicksum, GRB
import json

# load data
with open('data.json') as f:
    data = json.load(f)

# define the number of oils and months
I = len(data['hardness'])
M = len(data['buy_price'][0])

# define model
model = Model()

# define decision variables
refine = model.addVars(I, M, vtype=GRB.CONTINUOUS, name="refine")

# assuming other parts of the model like objective function and other constraints are defined here
# ...

# loop over each time period
for m in range(M):
    # add hardness constraints
    if sum(refine[i, m] for i in range(I)) > 0: # avoid division by zero
        model.addConstr(quicksum(data['hardness'][i]*refine[i, m] for i in range(I)) / quicksum(refine[i, m] for i in range(I)) >= data['min_hardness'])
        model.addConstr(quicksum(data['hardness'][i]*refine[i, m] for i in range(I)) / quicksum(refine[i, m] for i in range(I)) <= data['max_hardness'])
