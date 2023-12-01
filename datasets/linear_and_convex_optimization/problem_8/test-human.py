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
    for key in ["batches", "total_profit"]:
        if key not in output:
            error_list.append(f"The output field '{key}' is missing")

    if not isinstance(output["batches"], list):
        error_list.append("The output field 'batches' should be a list")

    if not isinstance(output["total_profit"], (int, float)):
        error_list.append("The value of 'total_profit' is not a number")

    if len(output["batches"]) != len(data["prices"]):
        error_list.append(
            "The output field 'batches' should have the same length as the 'prices' list in the input"
        )

    if any(x < 0 for x in output["batches"]):
        error_list.append("All values in the 'batches' list should be non-negative")

    # Ensure machine time does not exceed availability
    machine_time_used = np.dot(
        np.array(data["time_required"]), np.array(output["batches"])
        )
    availability = np.array(data["availability"])
    total_availability = availability + np.array(output["extra_time"])
    for m, (used, available) in enumerate(zip(machine_time_used, total_availability)):
        if used - eps > available:
            error_list.append(f"Total time used on machine {m+1} exceeds available time.")

    # Ensure at least the minimum required batches are produced
    for p, (produced, required) in enumerate(zip(np.array(output["batches"]), np.array(data["min_batches"]))):
        if required - produced > eps:
            error_list.append(
                f"The total batches for part {p+1} is less than the requirement"
            )

    # Ensure extra time does not exceed max
    for m, extra_time in enumerate(output["extra_time"]):
        if extra_time - data["max_extra"][m] > eps:
            error_list.append(f"Extra time purchased for machine {m+1} exceeds the maximum allowed.")

    # Ensure total profit is calculated correctly
    total_revenue = np.sum(np.array(output["batches"]) * np.array(data["prices"]))
    total_cost = np.sum(machine_time_used * np.array(data["machine_costs"])) + np.sum(np.array(output["extra_time"]) * np.array(data["extra_costs"]))
    if abs((total_revenue - total_cost) - output["total_profit"]) > eps:
        error_list.append(
            f"The total profit is not correctly calculated"
        )
    return error_list


if __name__ == "__main__":
    print(run())
