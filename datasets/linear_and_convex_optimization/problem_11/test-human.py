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
    for key in ["distribution", "total_cost"]:
        if key not in output:
            error_list.append(f"The output field '{key}' is missing")

    # Extracting necessary info
    route_costs = {(route["start"], route["end"]): route["C"] for route in data["routes"]}
    supply = {terminal["terminal"]: terminal["S"] for terminal in data["supply"]}
    demand = {dest["destination"]: dest["D"] for dest in data["demand"]}

    total_sent_to_port = {k: 0 for k in supply}
    total_received_at_port = {p["end"]: 0 for p in data["routes"] if p["end"] not in [d["destination"] for d in data["demand"]]}
    total_sent_from_port = {p["start"]: 0 for p in data["routes"] if p["start"] not in [s["terminal"] for s in data["supply"]]}
    total_received_at_destination = {l: 0 for l in demand}
    calculated_cost = 0

    for distribution in output["distribution"]:
        i, j = distribution["from"], distribution["to"]
        # Check  amounts for all routes
        if (i, j) not in route_costs:
            error_list.append(f"Unexpected route in output: ({i}, {j})")
            continue
        # Check if 'i' is a terminal
        if i in [s["terminal"] for s in data["supply"]]:
            total_sent_to_port[i] += distribution["amount"]
        else:  # 'i' is a port
            total_sent_from_port[i] += distribution["amount"]
        # Check if 'j' is a destination
        if j in [d["destination"] for d in data["demand"]]:
            total_received_at_destination[j] += distribution["amount"]
        else:
            total_received_at_port[j] += distribution["amount"]
        calculated_cost += distribution["amount"] * route_costs[(i, j)]
        
    # Check Terminal to Port flow constraints
    for k, sent in total_sent_to_port.items():
        if sent - supply[k] > eps:
            error_list.append(f"Flow constraint violated for terminal {k}. Sent: {sent}, Supply: {supply[k]}")

    # Check Demand constraints
    for l, received in total_received_at_destination.items():
        demand_for_l = next(d["D"] for d in data["demand"] if d["destination"] == l)
        if demand_for_l - received > eps:
            error_list.append(f"Demand not met for destination {l}. Received: {received}, Demand: {demand_for_l}")
    
    # Check Port flow conservation
    for p in total_received_at_port:
        if abs(total_received_at_port[p] - total_sent_from_port.get(p, 0)) > eps:
            error_list.append(f"Flow conservation violated at port {p}. Received: {total_received_at_port[p]}, Sent: {total_sent_from_port.get(p, 0)}")

    # Check 5: Total cost verification
    if abs(calculated_cost - output["total_cost"]) > eps:
        error_list.append(f"Total cost mismatch. Calculated: {calculated_cost}, Given in output: {output['total_cost']}")
    return error_list


if __name__ == "__main__":
    print(run())
