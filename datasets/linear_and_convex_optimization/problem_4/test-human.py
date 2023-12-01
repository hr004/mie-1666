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
    for key in ["alloy_use", "total_steel", "total_profit"]:
        if key not in all_json_keys:
            error_list.append(f"The output field '{key}' is missing")

    # Check the types of the output keys.
    if not isinstance(output["alloy_use"], list):
        error_list.append("The output field 'alloy_use' should be a list")

    if not isinstance(output["total_steel"], list):
        error_list.append("The output field 'total_steel' should be a list")

    if not isinstance(output["total_profit"], (int, float)):
        error_list.append("The value of 'total_profit' is not a number")

    # Check if the length of the alloy_use in output matches the length of costs in data.
    if len(output["alloy_use"]) != len(data["steel_prices"]):
        error_list.append(
            "The output field 'alloy_use' should have the same length as the 'steel_prices' list in the input"
        )

    if len(output["total_steel"]) != len(data["steel_prices"]):
        error_list.append(
            "The output field 'total_steel' should have the same length as the 'steel_prices' list in the input"
        )

    # Check all values in the alloy_use list are non-negative.
    for s in range(len(data["steel_prices"])):
        for a in range(len(data["available"])):
            if output["alloy_use"][s][a] < 0:
                error_list.append("All values in the 'alloy_use' list should be non-negative")
                break

    total_profit_calculated = 0

    for s in range(len(data["steel_prices"])):
        total_carbon = 0
        total_nickel = 0
        total_cost = 0
        for a in range(len(data["available"])):
            total_carbon += output["alloy_use"][s][a] * data["carbon"][a] / 100.0
            total_nickel += output["alloy_use"][s][a] * data["nickel"][a] / 100.0
            total_cost += output["alloy_use"][s][a] * data["alloy_prices"][a]

            # Validate the 40% constraint for alloy 1
            if a == 0 and s == 0 and output["alloy_use"][s][a] > 0.4 * output["total_steel"][s] + eps:
                error_list.append(f"Steel {s+1} contains more than 40% of alloy 1")

        # Validate minimum carbon and maximum nickel requirements
        if total_carbon < data["carbon_min"][s] * output["total_steel"][s] / 100.0 - eps:
            error_list.append(f"Steel {s+1} does not meet the minimum carbon requirement")
        if total_nickel > data["nickel_max"][s] * output["total_steel"][s] / 100.0 + eps:
            error_list.append(f"Steel {s+1} exceeds the maximum nickel percentage")

        # Calculate total profit
        profit_for_s = (data["steel_prices"][s] * output["total_steel"][s]) - total_cost
        total_profit_calculated += profit_for_s

    # Validate if alloy used does not exceed the available amount
    for a in range(len(data["available"])):
        alloy_used = sum([output["alloy_use"][s][a] for s in range(len(data["steel_prices"]))])
        if alloy_used > data["available"][a] + eps:
            error_list.append(f"The total amount of alloy {a+1} used exceeds the available amount")

    # Validate profit calculation
    if abs(total_profit_calculated - output["total_profit"]) > eps:
        error_list.append(f"The total profit is not correctly calculated'")
    return error_list


if __name__ == "__main__":
    print(run())
