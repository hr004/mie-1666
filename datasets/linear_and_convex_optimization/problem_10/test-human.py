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
    for key in ['send', 'total_cost']:
        if key not in output:
            error_list.append(f"The output field '{key}' is missing")

    if not isinstance(output['send'], list) or not all(isinstance(plant_output, list) for plant_output in output['send']):
        error_list.append("The output field 'send' should be a list")

    if (len(output["send"]) != len(data["transmission_costs"])) or (len(output["send"][0]) != len(data["transmission_costs"][0])):
        error_list.append(
            "The output field 'send' should have the same length as the 'transmission_costs' list in the input"
        )

    # Check all values in the send list are non-negative.
    for p_idx, plant_output in enumerate(output['send']):
        for c_idx, sent_value in enumerate(plant_output):
            if sent_value < 0:
                error_list.append("All values in the 'send' list should be non-negative")
                break

    # Check if the total electricity sent by each plant does not exceed its supply
    for p_idx, plant_output in enumerate(output['send']):
        if sum(plant_output) > data['supply'][p_idx] + eps:
            error_list.append(f"Plant {p_idx + 1} exceeded its supply. Allowed: {data['supply'][p_idx]}, Sent: {sum(plant_output)}")

    # Check if the total electricity received by each city matches its demand
    for c_idx in range(len(data['demand'])):
        city_received = sum([plant_output[c_idx] for plant_output in output['send']])
        if abs(city_received - data['demand'][c_idx]) > eps:
            error_list.append(f"City {c_idx + 1} demand mismatch. Expected: {data['demand'][c_idx]}, Received: {city_received}")

    # Check if the total cost in the output is equal to the sum of the product of electricity sent and the transmission cost for each route
    calculated_cost = np.sum(np.multiply(data['transmission_costs'], output['send']))
    if abs(calculated_cost - output['total_cost']) > eps:
        error_list.append(f"The total cost is not correctly calculated")
    return error_list


if __name__ == "__main__":
    print(run())
