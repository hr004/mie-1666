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
    for key in ["officers_assigned", "total_cost"]:
        if key not in all_json_keys:
            error_list.append(f"The output field '{key}' is missing")

    # Check the types of the output keys.
    if not isinstance(output["officers_assigned"], list):
        error_list.append("The output field 'officers_assigned' should be a list")

    if not isinstance(output["total_cost"], (int, float)):
        error_list.append("The output field 'total_cost' should be a number")

    # Check if the length of the alloy_use in output matches the length of costs in data.
    if len(output["officers_assigned"]) != len(data["officers_needed"]):
        error_list.append(
            "The output field 'officers_assigned' should have the same length as the 'officers_needed' list in the input"
        )

    # Check all values in the alloy_use list are non-negative.
    if any(x < 0 for x in output["officers_assigned"]):
        error_list.append("All values in the 'officers_assigned' list should be non-negative")

    # Check the number of officers assigned
    for i in range(len(data["officers_needed"])):
        required_officers = data["officers_needed"][i]
        assigned_officers = output["officers_assigned"][i-1] + output["officers_assigned"][i]
        if required_officers - assigned_officers > eps:
            error_list.append(f"Shift {i+1} requires {required_officers} officers but {assigned_officers} were assigned.")

    # Check the total cost
    calculated_cost = np.dot(
        data["shift_costs"], output["officers_assigned"]
        )
    if abs(calculated_cost - output["total_cost"]) > eps:
        error_list.append(f"The total cost is not correctly calculated")
    return error_list


if __name__ == "__main__":
    print(run())
