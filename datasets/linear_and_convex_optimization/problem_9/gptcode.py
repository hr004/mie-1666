Your task is to formulate and solve the given optimization problem as a LP. Please read the problem information, input format, and objective carefully and provide a detailed mathematical formulation.

### PROBLEM INFORMATION:


- Custom Tees is planning an online advertising campaign with \var{A} different ad types across two web companies.
- The company has set a goal of \var{goal_young} thousand clicks from visitors aged 18-25 and \var{goal_old} thousand clicks from visitors older than 25.
- The company has set a goal of \var{goal_unique_young} thousand unique clicks from visitors aged 18-25 and \var{goal_unique_old} thousand unique clicks from visitors older than 25.
- Ad type \var{a} has an estimated percentage of \var{young_click_{a}} clicks from the 18-25 age range.
- Ad type \var{a} has an estimated percentage of \var{old_click_{a}} clicks from older visitors.
- Ad type \var{a} has a cost of \var{cost_{a}} per 1000 clicks.
- Ad type \var{a} has a maximum allowable clicks of \var{max_click_{a}} in thousands.
- The estimated percentage of unique visitors for each ad type \var{a} is \var{unique_click_{a}}.
- The company has an advertising budget of $\var{budget}


### INPUT FORMAT:


{
    "goal_young": goal_young,
    "goal_old": goal_old,
    "goal_unique_young": goal_unique_young,
    "goal_unique_old": goal_unique_old,
    "young_clicks": [young_click_{a} for a in 1,...,A],
    "old_clicks": [old_click_{a} for a in 1,...,A],
    "costs": [cost_{a} for a in 1,...,A],
    "max_clicks": [max_click_{a} for a in 1,...,A],
    "unique_clicks": [unique_click_{a} for a in 1,...,A],
    "budget": budget
}


- Variables enclosed in [ ] represent lists of values.
- Names enclosed in quotes (") represent keys and are identical to those in the data file.
- All other items are variables as described in the problem description and should be replaced with their actual values from the data file.

### OBJECTIVE: 

How many clicks from each ad type should Custom Tees purchase to maximize the number of unique clicks within the budget?

### OUTPUT INFORMATION:


- the number of clicks purchased for each ad type is \var{clicks_{a}} in thousands.
- \var{total_unique_clicks} represents the total number of unique clicks.


### OUTPUT FORMAT:


{
    "clicks": [click_{a} for a in 1, ..., A],
    "total_unique_clicks": total_unique_clicks
}


### INSTRUCTIONS:
1. Clearly define the decision variables.
2. Formulate the objective function precisely.
3. List all the constraints, ensuring they are complete and non-redundant.
4. Ensure the formulation is coherent, logical, and solvable.
5. Provide any necessary explanations or clarifications for your formulation.


Please respond with a well-structured mathematical formulation of the given optimization problem, adhering to the instructions and format provided above.
### DECISION VARIABLES:

Let's define the decision variable as:
- \( x_{a} \) = number of clicks Custom Tees purchases from ad type \( a \) (in thousands)

### OBJECTIVE FUNCTION:

The company wants to maximize the number of unique clicks. The total number of unique clicks is the sum of the unique clicks from each ad type, which is the product of the number of clicks purchased and the percentage of unique clicks for that ad type. So, the objective function is:

Maximize \( Z = \sum_{a=1}^{A} x_{a} * unique\_click_{a} \)

### CONSTRAINTS:

1. **Budget constraint**: The total cost of the purchased clicks must not exceed the company's budget. The cost of the clicks from each ad type is the product of the number of clicks purchased and the cost per thousand clicks for that ad type. So, the constraint is:

   \( \sum_{a=1}^{A} x_{a} * cost_{a} \leq budget \)

2. **Young age group click goal constraint**: The total clicks from the young age group (18-25 years) must be at least the goal set by the company. The clicks from the young age group from each ad type is the product of the number of clicks purchased and the estimated percentage of clicks from the young age group for that ad type. So, the constraint is:

   \( \sum_{a=1}^{A} x_{a} * young\_click_{a} \geq goal\_young \)

3. **Old age group click goal constraint**: The total clicks from the old age group (over 25 years) must be at least the goal set by the company. The clicks from the old age group from each ad type is the product of the number of clicks purchased and the estimated percentage of clicks from the old age group for that ad type. So, the constraint is:

   \( \sum_{a=1}^{A} x_{a} * old\_click_{a} \geq goal\_old \)

4. **Young age group unique click goal constraint**: Similarly, the total unique clicks from the young age group must be at least the goal set by the company. So, the constraint is:

   \( \sum_{a=1}^{A} x_{a} * young\_click_{a} * unique\_click_{a} \geq goal\_unique\_young \)

5. **Old age group unique click goal constraint**: The total unique clicks from the old age group must be at least the goal set by the company. So, the constraint is:

   \( \sum_{a=1}^{A} x_{a} * old\_click_{a} * unique\_click_{a} \geq goal\_unique\_old \)

6. **Maximum allowable clicks constraint**: The number of clicks purchased from each ad type must not exceed the maximum allowable clicks for that ad type. So, the constraints are:

   \( x_{a} \leq max\_click_{a} \) for all \( a \)

7. **Non-negativity constraint**: The number of purchased clicks cannot be negative.

   \( x_{a} \geq 0 \) for all \( a \)

This completes the mathematical formulation of the given LP problem.I am using a None code in python to formulate and solve this problem. The code, provided below, reads data from a file named "data.json" and writes the output to a file named "output.json". However, a user has reported that the output is incorrect and provided a list of error messages.

### CODE: 


import json
from gurobipy import *

# Read data
with open('data.json', 'r') as f:
    data = json.load(f)

# Define model
model = Model('CustomTees_AdCampaign')

# Add decision variables
x = model.addVars(len(data['costs']), vtype=GRB.CONTINUOUS, name="x")

# Set objective function
model.setObjective(sum(x[a] * data['unique_clicks'][a] for a in range(len(data['costs']))), GRB.MAXIMIZE)

# Add budget constraint
model.addConstr(sum(x[a] * data['costs'][a] for a in range(len(data['costs']))) <= data['budget'], "budget")

# Add young age group click goal constraint
model.addConstr(sum(x[a] * data['young_clicks'][a] for a in range(len(data['costs']))) >= data['goal_young'], "young_click_goal")

# Add old age group click goal constraint
model.addConstr(sum(x[a] * data['old_clicks'][a] for a in range(len(data['costs']))) >= data['goal_old'], "old_click_goal")

# Add young age group unique click goal constraint
model.addConstr(sum(x[a] * data['young_clicks'][a] * data['unique_clicks'][a] for a in range(len(data['costs']))) >= data['goal_unique_young'], "young_unique_click_goal")

# Add old age group unique click goal constraint
model.addConstr(sum(x[a] * data['old_clicks'][a] * data['unique_clicks'][a] for a in range(len(data['costs']))) >= data['goal_unique_old'], "old_unique_click_goal")

# Add maximum allowable clicks constraint
model.addConstrs((x[a] <= data['max_clicks'][a] for a in range(len(data['costs']))), "max_clicks")

# Solve model
model.optimize()

if model.status == GRB.Status.OPTIMAL:
    # Save results
    output = {
        "clicks": [x[a].x for a in range(len(data['costs']))],
        "total_unique_clicks": sum(x[a].x * data['unique_clicks'][a] for a in range(len(data['costs']))),
        "total_cost": sum(x[a].x * data['costs'][a] for a in range(len(data['costs'])))
    }

    # Added the missing lines in the original code to fix the issue.
    with open('output.json', 'w') as f:
        json.dump(output, f)


### ERROR MESSAGE: 

['The total cost in the output (70300.0) is not equal to the calculated total distance (703.0)']

### YOUR TASK:

1. Could you please provide a detailed explanation of each error mentioned in the ['The total cost in the output (70300.0) is not equal to the calculated total distance (703.0)']?
2. Could you identify the sections of the code that are likely causing these errors?
3. Could you suggest modifications to the code to resolve the errors and fix the output?

Based on the above information, please generate a corrected version of the entire code.

