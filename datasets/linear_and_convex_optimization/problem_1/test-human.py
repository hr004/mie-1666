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

    # Check if all the required keys are present in the output.
    for key in ["amount", "total_profit"]:
        if key not in output:
            error_list.append(f"The output field '{key}' is missing")

    if not isinstance(output["amount"], list):
        error_list.append("The output field 'amount' should be a list")

    if len(output["amount"]) != len(data["prices"]):
        error_list.append(
            "The output field 'amount' should have the same length as the 'prices' list in the input"
        )

    if any(x < 0 for x in output["amount"]):
        error_list.append("All values in the 'amount' list should be non-negative")

    total_material_used = np.dot(
        np.array(data["requirements"]).T, np.array(output["amount"])
    )
    for i, (used, available) in enumerate(zip(total_material_used, data["available"])):
        if used - available > eps:
            error_list.append(
                f"The total amount of raw material {i+1} used exceeds the available amount"
            )

    for i, (amount, demand) in enumerate(zip(output["amount"], data["demands"])):
        if amount - demand > eps:
            error_list.append(
                f"The total amount of product {i+1} exceeds the forecasted maximum demand"
            )

    total_price = np.dot(
        np.array(data["prices"]).T, np.array(output["amount"])
    )
    total_cost = np.dot(
        np.array(data["costs"]).T, np.array(output["amount"])
    ) 
    if abs((total_price - total_cost) - output["total_profit"]) > eps:
        error_list.append(
            f"The total profit is not correctly calculated"
        )
    return error_list


if __name__ == "__main__":
    print(run())
