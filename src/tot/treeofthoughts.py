# From tree-of-thoughts lib

import concurrent.futures
import json
import logging
import os
from typing import Any, Dict, Union

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TreeofThoughts:
    def __init__(self, model):
        self.model = model
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = []  # added line initalize history

    def save_tree_to_json(self, file_name):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as json_file:
            json.dump(self.tree, json_file, indent=4)

    def logNewState(self, state, evaluation):
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree["nodes"]:
            self.tree["nodes"][state]["thoughts"].append(evaluation)
        else:
            self.tree["nodes"][state] = {"thoughts": [evaluation]}

    def adjust_pruning_threshold_precentile(self, evaluated_thoughts, percentile):
        values = np.array(list(evaluated_thoughts.values()))
        if values.size == 0:
            return 0
        return max(np.percentile(values, percentile), 0.1)

    def adjust_pruning_threshold_moving_average(self, evaluated_thoughts, window_size):
        values = list(evaluated_thoughts.values())
        if len(values) < window_size:
            return np.mean(values) if values else 0
        else:
            return max(np.mean(values[-window_size:]), 0.1)


class TreeofThoughtsBFS(TreeofThoughts):
    def solve(
        self,
        initial_prompt,
        num_thoughts,
        max_steps,
        max_states,
        value_threshold,
        pruning_threshold=0.5,
    ):
        current_states = [initial_prompt]
        state_values = {}
        dynamic_pruning_threshold = pruning_threshold

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for step in range(1, max_steps + 1):
                    selected_states = []
                    for state in current_states:
                        thoughts = self.model.generate_thoughts(
                            state, num_thoughts, initial_prompt
                        )
                        futures = [
                            executor.submit(
                                self.model.evaluate_states,
                                {thought: 0},
                                initial_prompt,
                            )
                            for thought in thoughts
                        ]
                        concurrent.futures.wait(futures)

                        evaluated_thoughts = {}
                        # evaluated_thoughts = {
                        #     thought: fut.result()
                        #     for thought, fut in zip(thoughts, futures)
                        # if isinstance(fut.result(), (int, float))
                        # }  # check if result is a number
                        for thought, fut in zip(thoughts, futures):
                            evaluated_thoughts[thought] = fut.result()[thought]

                        if (
                            evaluated_thoughts
                        ):  # only adjust if you have evaluated thoughts
                            dynamic_pruning_threshold = (
                                self.adjust_pruning_threshold_moving_average(
                                    evaluated_thoughts, 5
                                )
                            )

                        for thought, value in evaluated_thoughts.items():
                            flattened_state = (
                                (state, thought)
                                if isinstance(state, str)
                                else (*state, thought)
                            )
                            selected_states.append((flattened_state, value))

                        selected_states.sort(key=lambda x: x[1], reverse=True)
                        selected_states = selected_states[
                            :max_states
                        ]  # Select only the top states

                        for state, value in selected_states:
                            if value >= dynamic_pruning_threshold:
                                state_values[state] = value
                                self.logNewState(state, value)
                                logger.debug(f"State Values: {state_values}")

            if state_values:
                highest_rated_solution = max(state_values.items(), key=lambda x: x[1])
                highest_rated_state = highest_rated_solution[0]
                solution = self.model.generate_solution(
                    initial_prompt, highest_rated_state
                )
                print(
                    "Highest_rated solution:"
                    f" {highest_rated_solution} highest_rated_solution:"
                    f" {highest_rated_solution} Solution: {solution}"
                )

                return solution if solution else highest_rated_state

            else:
                return None

        except Exception as e:
            logger.error(f"Error in tot_bfs: {e}")
            return None
