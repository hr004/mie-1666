from pathlib import Path
from src.problems import ProblemReader
from src.configure import AgentBehaviorParameters
from src.agent import OptiMus
from src.messages import Messages
from src.gpt_or import GptOr
from src.configure import MODE_COT_HUMAN


def main():
    path = Path("./datasets/lectures_in_lp_modeling/problem_7").absolute()
    problem_description_file = "description.txt"
    benchmark_file = "test-human.py"
    log_file = "description-{cot_mode}.log"

    problem = ProblemReader(
        path,
        problem_description_file,
        benchmark_file,
        log_file.format(cot_mode=MODE_COT_HUMAN),
    )

    agent_params = AgentBehaviorParameters()
    conversations = Messages(problem=problem)
    gpt_or = GptOr(agent_params=agent_params, conversations=conversations)

    optimus = OptiMus(gpt_or=gpt_or)
    optimus.solve_problem()


if __name__ == "__main__":
    main()
