

import os
import subprocess
import importlib
from src.gpt_or import GptOr


class OptiMus:
    def __init__(self, gpt_or: GptOr) -> None:
        self.gpt_or = gpt_or
        self.solver_codes = []

    def solve_problem(self):
        formulation_response = self.gpt_or.generate_problem_formulation()
        code_generation_response = self.gpt_or.generate_problem_code()
        try:
            self.solver_codes.append(code_generation_response.split("```")[1][6:])
        except IndexError:
            self.solver_codes.append(code_generation_response)

        self.gpt_or.conversations.problem.data["code"] = self.solver_codes[-1]
        self.dump_code()

        success = self.run()
        if success:
            print("Hurray, solved the problem")
        else:
            self.fix()

    # create two funcs. run and fix
    def run(self):
        # execute gpt code and human test file
        success = False
        self.gpt_or.conversations.problem.human_test_file
        execution_ok, error = self.execute_generated_code()
        if execution_ok:
            test = self.load_test_file()
            test_result_error = self.execute_test_code(test)
            if len(test_result_error) == 0:
                success = True
        if not execution_ok:
            # code fix
            pass
        return success

    def fix(self):
        pass

    def execute_test_code(self, test):
        # TODO: syntax only if else
        result = []
        try:
            result = test.run()
        except Exception as e:
            print("Test script is invalid! something wrong in output.json")
        return result

    def execute_generated_code(self):
        error = None
        execution_ok = False
        try:
            os.chdir(self.gpt_or.conversations.problem.path_to_gpt_code.parent)
            # this will create output.json and test-human will read that file and make sure the output is correct or not
            subprocess.run(
                [
                    "python",
                    self.gpt_or.conversations.problem.path_to_gpt_code
                ],
                check= True,
                text=True,
                capture_output=True
            )

            execution_ok = True
        except subprocess.CalledProcessError as e:
            error = e.stderr
            print("Script failed and exited with an error code: %s" % e.returncode)
        finally:
            return execution_ok, error

    def load_test_file(self):
        spec = importlib.util.spec_from_file_location(
            "test",
            self.gpt_or.conversations.problem.human_test_file
            )
        test = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test)
        return test

    def dump_code(self):
        if self.gpt_or.conversations.problem.data["code"] is None:
            raise ValueError("No code to dump")
        with open(self.gpt_or.conversations.problem.path_to_gpt_code, "w") as f:
            f.write(self.solver_codes[-1])

    def dump_conversation(self):
        print("Dumping conversation")
        with open(self.gpt_or.conversations.path_to_conversation, "w") as f:
            f.write("\n".join(self.gpt_or.conversations.global_conversations))
