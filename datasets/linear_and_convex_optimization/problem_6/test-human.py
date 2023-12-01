import json
import numpy as np

eps = 1e-03


def parse_json(json_file, keys):
    if isinstance(json_file, list):
        for v in json_file:
            parse_json(v, keys)
    elif isinstance(json_file, dict):
        for k, v in json_file.items():
            if isinstance(v, dict) or isinstance(v, list):
                parse_json(v, keys)
            if k not in keys:
                keys.append(k)


def run():
    with open("data.json", "r") as f:
        data = json.load(f)

    with open("output.json", "r") as f:
        output = json.load(f)

    all_json_keys = []
    parse_json(output, all_json_keys)

    error_list = []

    # Check if all necessary keys are present in the output JSON.
    for key in ["regular_used", "overtime_used", "regular_baskets", "overtime_baskets", "inventory", "total_profit"]:
        if key not in all_json_keys:
            error_list.append(f"The output field '{key}' is missing")

    # Check the types of the output keys.
    for key in ["regular_used", "overtime_used", "regular_baskets", "overtime_baskets", "inventory"]:
        if not isinstance(output[key], list):
            error_list.append(f"The output field '{key}' should be a list")

    if not isinstance(output["total_profit"], (int, float)):
        error_list.append("The output field 'total_profit' should be a number")

    # Check if the length of the output matches the length of costs in data.
    for key in ["regular_used", "overtime_used", "regular_baskets", "overtime_baskets", "inventory"]:
        if len(output[key]) != len(data["demand"]):
            error_list.append(
                f"The output field '{key}' should have the same length as the 'demand' list in the input"
            )

    # Check all values in the output list are non-negative.
    for key in ["regular_used", "overtime_used", "regular_baskets", "overtime_baskets", "inventory"]:
        if any(x < 0 for x in output[key]):
            error_list.append(f"All values in the '{key}' list should be non-negative")

    # Check labor constraints
    W = len(data["demand"])
    for w in range(W):
        if output["regular_used"][w] > data["regular_labor"][w] + eps:
            error_list.append(f"Week {w+1}: Exceeded regular labor hours.")
        if output["overtime_used"][w] > data["overtime_labor"][w] + eps:
            error_list.append(f"Week {w+1}: Exceeded overtime labor hours.")

    # Check basket assembling constraints
    for w in range(W):
        if output["regular_baskets"][w] > data["regular_labor"][w] / data["assembly_time"] + eps:
            error_list.append(f"Week {w+1}: Number of baskets assembled exceeds labor hours used.")
        if output["overtime_baskets"][w] > data["overtime_labor"][w] / data["assembly_time"] + eps:
            error_list.append(f"Week {w+1}: Number of baskets assembled exceeds labor hours used.")

    # Check inventory calculations
    prev_inventory = 0
    for w in range(W):
        expected_inventory = prev_inventory + output["regular_baskets"][w] + output["overtime_baskets"][w] - data["demand"][w]
        if abs(expected_inventory - output["inventory"][w]) > eps:
            error_list.append(f"Week {w+1}: Inventory value is incorrect.")
        prev_inventory = output["inventory"][w]

    # Check profit calculations
    profit = 0
    for w in range(W):   #(demand, reg_baskets, ot_baskets, inventory_out, reg_used, ot_used) in enumerate(zip(data["demand"], output["regular_baskets"], output["overtime_baskets"], output["inventory"], output["regular_used"], output["overtime_used"])):
        sold = output["regular_baskets"][w] + output["overtime_baskets"][w]
        holding_cost = output["inventory"][w] * data["holding_cost"] if w != W - 1 else 0
        salvage_value = output["inventory"][w] * (data["selling_price"] - data["salvage_value"]) if w == W - 1 else 0
        labor_cost = output["regular_used"][w] * data["regular_cost"] + output["overtime_used"][w] * data["overtime_cost"]
        profit += (data["selling_price"] - data["material_cost"]) * sold - labor_cost - holding_cost - salvage_value 
    if abs(profit - output["total_profit"]) > eps:
        error_list.append(f"Calculated profit ({profit}) does not match given total profit ({output['total_profit']}).")

    return error_list


if __name__ == "__main__":
    print(run())
