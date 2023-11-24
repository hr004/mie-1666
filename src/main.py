from pathlib import Path
from src.problems import ProblemReader
from src.configure import AgentBehaviorParameters
from src.agent import OptiMus
from src.messages import Messages
from src.gpt_or import GptOr
from src.configure import MODE_COT_HUMAN
import json

# sub_dataset = "introduction_to_linear_optimization"
# sub_dataset = "lectures_in_lp_modeling"
sub_dataset = "model_building_in_mathematical_programming"

def main(abs_path: Path):
    print(abs_path)
    problem_description_file = "description.txt"
    benchmark_file = "test-human.py"
    log_file = "description-{cot_mode}.log"

    problem = ProblemReader(
        abs_path,
        problem_description_file,
        benchmark_file,
        log_file.format(cot_mode=MODE_COT_HUMAN),
    )

    agent_params = AgentBehaviorParameters()
    conversations = Messages(problem=problem)
    gpt_or = GptOr(agent_params=agent_params, conversations=conversations)

    optimus = OptiMus(gpt_or=gpt_or)
    is_solved, total_attempts = optimus.solve_problem()
    return is_solved, total_attempts

def run_all():
    import os 
    results = {}
    directory = f"{os.getcwd()}/datasets/{sub_dataset}/"
    problems = [p for p in os.listdir(directory) if p.startswith("problem")]
    # paths = [Path(f"{os.getcwd()}/datasets/introduction_to_linear_optimization/problem_{i}") for i in range(1,12)]
    # print(paths)
    for i, problem in enumerate(problems):
        path = Path(f"{directory}/{problem}").absolute()
        print(f"Solving problem {path}")
        is_solved, total_attempts = main(path)
        results[f"lp_modelling_{i}"] = {
            "success": is_solved,
            "attempts": total_attempts
        }
    return results

if __name__ == "__main__":
    results = run_all()
    json.dump(results, open(f"./results/{sub_dataset}.json", "w"))
