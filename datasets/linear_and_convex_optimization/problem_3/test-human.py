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
    required_keys = ["clicks", "total_cost"]
    for key in required_keys:
        if key not in all_json_keys:
            error_list.append(f"The output field '{key}' is missing")

    # Check the types of the output keys.
    if not isinstance(output["clicks"], list):
        error_list.append("The output field 'clicks' should be a list")

    if not isinstance(output["total_cost"], (int, float)):
        error_list.append("The value of 'total_cost' is not a number")

    # Check if the length of the clicks in output matches the length of costs in data.
    if len(output["clicks"]) != len(data["costs"]):
        error_list.append(
            "The output field 'clicks' should have the same length as the 'costs' list in the input"
        )

    # Check all values in the clicks list are non-negative.
    if any(x < 0 for x in output["clicks"]):
        error_list.append("All values in the 'clicks' list should be non-negative")

     # Check each ad type's clicks against their respective maximum allowable clicks.
    for i, (click, max_click) in enumerate(zip(output["clicks"], data["max_clicks"])):
        if click > max_click + eps:
            error_list.append(f"Ad type {i + 1} exceeds maximum allowable clicks.")

    # Check total young and old clicks against the set goals.
    total_young_clicks = sum(click * perc / 100.0 for click, perc in zip(output["clicks"], data["young_clicks"]))
    total_old_clicks = sum(click * perc / 100.0 for click, perc in zip(output["clicks"], data["old_clicks"]))

    if total_young_clicks - data["goal_young"] < -eps:
        error_list.append(f"Total young clicks {total_young_clicks} do not match the goal {data['goal_young']}.")
    if total_old_clicks - data["goal_old"] < -eps:
        error_list.append(f"Total old clicks {total_old_clicks} do not match the goal {data['goal_old']}.")

    # Check total unique young and old clicks against the set goals.
    total_unique_young_clicks = sum(click * perc / 100.0 * unique / 100.0 for click, perc, unique in zip(output["clicks"], data["young_clicks"], data['unique_clicks']))
    total_unique_old_clicks = sum(click * perc / 100.0 * unique / 100.0 for click, perc, unique in zip(output["clicks"], data["old_clicks"], data['unique_clicks']))

    if total_unique_young_clicks - data["goal_unique_young"] < -eps:
        error_list.append(f"Total unique young clicks {total_unique_young_clicks} do not match the goal {data['goal_unique_young']}.")
    if total_unique_old_clicks - data["goal_unique_old"] < -eps:
        error_list.append(f"Total unique old clicks {total_unique_old_clicks} do not match the goal {data['goal_unique_old']}.")

    # Check the calculated total cost against the provided total cost in the output.
    total_cost = sum(click * cost for click, cost in zip(output["clicks"], data["costs"]))
    if abs(total_cost - output["total_cost"]) > eps:
        error_list.append(f"The total cost in the output ({output['total_cost']}) is not equal to the calculated total distance ({total_cost})")

    return error_list


if __name__ == "__main__":
    print(run())
