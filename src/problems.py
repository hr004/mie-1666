
import os
from src.utils import get_initial_test_script, read_problem_from_entire_file

class ProblemReader:
    def __init__(self, problem_path, problem_description_file, benchmark_file, log_file) -> None:
        self.problem_path: str = problem_path
        # self.std_format: str = None
        # std_format hocche description file
        self.problem_description_file: str = problem_description_file
        self.benchmark_file: str = benchmark_file
        self.log_file: str = log_file

        self.data = self.get_problem_data()

    def get_problem_data(self):
        data = self.read_problem_from_entire_file()

        initial_test_script = get_initial_test_script(
            output_format=data["output_format"]
        )

        data["initial_test_script"] = initial_test_script
        data["code_available"] = ""

        return data

    @property
    def solver(self):
        if self.data["problem_type"] in ["MILP", "ILP"]:
            return "gurobi"

    @property
    def errmsg(self):
        return ""

    def read_problem_from_entire_file(self):
        data = read_problem_from_entire_file(
            os.path.join(self.problem_path, self.problem_description_file)
        )
        return data
    
    @property
    def path_to_gpt_code(self):
        return self.problem_path.joinpath("gptcode.py")

    @property
    def human_test_file(self):
        return self.problem_path.joinpath("test-human.py")
