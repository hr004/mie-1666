
import json

# define and initialize spacegrain list
spacegrain = [0 for i in range(5)]

# set the current year index
t = 0

# load data from the file
with open("data.json") as json_file:
    data = json.load(json_file)

    # calculate the expression
    result = data['sellprice_grain'] * spacegrain[t] * data['yield'][t] - data['buyprice_grain'] * (data['n_milk'] * data['cow_grain'] - spacegrain[t] * data['yield'][t]) - data['extra_grain'] * spacegrain[t] - data['labour_grain'] * spacegrain[t] * data['extra_labour_cost']

    # print the result
    print(result)
