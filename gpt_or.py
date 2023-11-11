"""
Implement the interface of applying GPT-based model to solve OR models
"""

import os
import subprocess
import importlib
import argparse
from typing import Optional
from configure import internal_prompt
from configure import (
    template_formulation,
    template_codegen,
    template_codegen_var,
    template_codegen_constr,
    template_codegen_objsolve,
    template_codefix_execution,
    template_codefix_data,
    template_rephrase,
    template_testgen,
    template_standard_prompt,
)
import langchain
from configure import api_keys
from langchain.schema import Generation, BaseMessage, ChatResult, ChatGeneration
# Import behavior parameters
from configure import (
    MODE_COT_ONLY,
    MODE_COT_DEBUG,
    MODE_COT_DEBUG_TEST,
    MODE_COT_HUMAN,
)
from pathlib import Path

# Import status code
from configure import (
    STATUS_PASSED,
    STATUS_SYNTAX_ERROR,
    STATUS_LOGIC_ERROR,
)

from typing import List, Dict

from langchain.prompts.chat import HumanMessagePromptTemplate, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage
from langchain.chat_models import ChatOpenAI

from utils import (
    read_problem_from_entire_file,
    get_initial_test_script,
    generate_instance_template,
    get_solver_instruction,
    get_solver_demo,
)


from langchain.cache import InMemoryCache, SQLiteCache
# langchain.llm_cache = InMemoryCache()
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

class AgentBehaviorParameters:
    def __init__(self, solve_params: Optional[Dict[str, bool]] = None) -> None:
        if solve_params:
            self.params = solve_params
        else:
            self.params = {
                "COT": True,
                "Debug": False,
                "Test": False,
                "Human": False
            }

    # rename it to update params based on solve_mode
    def check_params(self, solve_mode):
        if solve_mode >= MODE_COT_ONLY:
            self.params["COT"] = True
        if solve_mode >= MODE_COT_DEBUG:
            self.params["Debug"] = True
        if solve_mode >= MODE_COT_DEBUG_TEST:
            self.params["Test"] = True
        if solve_mode >= MODE_COT_HUMAN:
            self.params["Human"] = True
            self.params["Test"] = False


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

class Conversations:
    def __init__(self, problem) -> None:
        self.global_conversations = []
        self.problem = problem
        self.system_message = SystemMessage(content=internal_prompt)

    def user_says_header(self):
        self.global_conversations.append("\n-------")
        self.global_conversations.append("User says: ")
        self.global_conversations.append("---------\n")

    def chatbot_says_header(self, model):
        self.global_conversations.append("\n----------")
        self.global_conversations.append(f"{model} says: ")
        self.global_conversations.append("------------\n")

    def prompt_format(self, template):
        chat_prompt_template = ChatPromptTemplate.from_messages(messages=template).format_messages(
            PROBLEM_TYPE=self.problem.data["problem_type"],
            PROBLEM_INFO=self.problem.data["problem_info"],
            INPUT_FORMAT=self.problem.data["input_format"],
            OBJECTIVE=self.problem.data["objective_info"],
            OUTPUT_INFO=self.problem.data["output_info"],
            OUTPUT_FORMAT=self.problem.data["output_format"],
            INITIAL_TEST_SCRIPT=self.problem.data["initial_test_script"],
            CODE=self.problem.data["code"],
            CODE_AVAILABLE=self.problem.data["code_available"],
            SOLVER=self.problem.solver,
            SOLVER_INSTRUCTION=get_solver_instruction(self.problem.solver),
            SOLVER_VAR_DEMO=get_solver_demo(self.problem.solver)["var"],
            SOLVER_CONSTR_DEMO=get_solver_demo(self.problem.solver)["constr"],
            SOLVER_SOLVE_DEMO=get_solver_demo(self.problem.solver)["solve"],
            ERROR_MESSAGE=self.problem.errmsg,
        )
        return chat_prompt_template

    def model_format(self, template):
        # this is used only for logging
        formatted_template = template.format(
            PROBLEM_TYPE = self.problem.data["problem_type"],
            PROBLEM_INFO = self.problem.data["problem_info"],
            INPUT_FORMAT=self.problem.data["input_format"],
            OBJECTIVE=self.problem.data["objective_info"],
            OUTPUT_INFO=self.problem.data["output_info"],
            OUTPUT_FORMAT=self.problem.data["output_format"],
            INITIAL_TEST_SCRIPT=self.problem.data["initial_test_script"],
            CODE=self.problem.data["code"],
            CODE_AVAILABLE=self.problem.data["code_available"],
            SOLVER=self.problem.solver,
            SOLVER_INSTRUCTION=get_solver_instruction(self.problem.solver),
            SOLVER_VAR_DEMO=get_solver_demo(self.problem.solver)["var"],
            SOLVER_CONSTR_DEMO=get_solver_demo(self.problem.solver)["constr"],
            SOLVER_SOLVE_DEMO=get_solver_demo(self.problem.solver)["solve"],
            ERROR_MESSAGE=self.problem.errmsg,
        )
        return formatted_template

    def get_formulation_conversation(self):
        formumation_request = HumanMessagePromptTemplate.from_template(template_formulation)

        conversation = [self.system_message, formumation_request]
        messages = self.prompt_format(conversation)

        # this is just to update conversation history
        # add update method
        self.user_says_header()
        self.global_conversations.append(self.system_message.content)
        self.global_conversations.append(self.model_format(template_formulation))
        self.chatbot_says_header("gpt-3.5-turbo")

        return messages


    def get_code_conversation(self):
        assert self.formulation_response is not None
        formulation_request = HumanMessagePromptTemplate.from_template(template_formulation)
        formulation_response = AIMessage(content=self.formulation_response)
        condegen_request = HumanMessagePromptTemplate.from_template(template_codegen)

        conversations = [
            self.system_message,
            formulation_request,
            formulation_response,
            condegen_request
        ]

        messages = self.prompt_format(conversations)

        self.user_says_header()
        self.global_conversations.append(self.system_message.content)
        self.global_conversations.append(self.model_format(template_codegen))
        self.chatbot_says_header("gpt-3.5-turbo")

        return messages

    def get_code_fix_conversation(self, execution_ok):
        formulation_request = HumanMessagePromptTemplate.from_template(
            template_formulation
        )
        formulation_response = AIMessage(content=self.formulation_response)
        codefix_execution_request = HumanMessagePromptTemplate.from_template(
            template_codefix_execution
        )
        codefix_data_request = HumanMessagePromptTemplate.from_template(
            template_codefix_data
        )

        if not execution_ok:
            codefix_request = codefix_execution_request
            template_codefix = template_codefix_execution
        else:
            codefix_request = codefix_data_request
            template_codefix = template_codefix_data

        conversations = [
            self.system_message,
            formulation_request,
            formulation_response,
            codefix_request
        ]

        messages = self.prompt_format(conversations)

        self.user_says_header()
        self.global_conversations.append(self.model_format(template_codefix))

        self.chatbot_says_header()

        return messages

    @property
    def formulation_response(self):
        return self._formulation_response

    @formulation_response.setter
    def formulation_response(self, llm_response: str):
        self._formulation_response = llm_response

    @property
    def path_to_conversation(self):
        return f"{self.problem.problem_path}/description.log"



# https://github.com/hwchase17/langchain/issues/1644
class CachedChatOpenAI(ChatOpenAI):
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None) -> ChatResult:
        messages_prompt = repr(messages)
        if langchain.llm_cache:
            results = langchain.llm_cache.lookup(messages_prompt, self.model_name)
            if results:
                assert len(results) == 1
                result: Generation = results[0]
                chat_result = ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=result.text))],
                    llm_output=result.generation_info)
                return chat_result
        chat_result = super()._generate(messages, stop)
        if langchain.llm_cache:
            assert len(chat_result.generations) == 1
            result = Generation(
                text=chat_result.generations[0].message.content,
                generation_info=chat_result.llm_output
            )
            langchain.llm_cache.update(messages_prompt, self.model_name, [result])
        return chat_result


class GPT4OR:
    def __init__(self, 
                 agent_params: AgentBehaviorParameters, 
                 conversations: Conversations
            ):
        self.agent_params = agent_params
        self.conversations = conversations
        self.llm: Optional[CachedChatOpenAI] = CachedChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model='gpt-3.5-turbo'
            )
        self.solver_codes = []

    def generate_problem_formulation(self):
        formulation_messages = self.conversations.get_formulation_conversation()
        output = self.llm(messages=formulation_messages)

        llm_response = output.content
        self.conversations.global_conversations.append(llm_response)
        self.conversations.formulation_response = llm_response
        return llm_response
        # print()

    def generate_problem_code(self):
        code_generation_messages = self.conversations.get_code_conversation()
        output = self.llm(messages=code_generation_messages)

        llm_response = output.content
        self.conversations.global_conversations.append(llm_response)
  
        return llm_response

    def generate_codefix_formulation(self):
        codefix_conversations = self.conversations.get_code_fix_conversation()
        output = self.llm(messages = codefix_conversations)
        llm_response = output.content
        self.conversations.global_conversations.append(llm_response)
        return llm_response

    def generate_test(self):
        pass

    def solve_problemm(self):
        pass

    @property
    def automatic_test(self):
        if self.agent_params["Debug"] and not self.agent_params["Test"] and not self.agent_params["Human"]:
            return True
        return False

class OptiMus:
    def __init__(self, gpt_or: GPT4OR) -> None:
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
        test = importlib.util.module_from_spec(spec)  # noqa
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


path = Path("./datasets/lectures_in_lp_modeling/problem_7").absolute()
problem_description_file = "description.txt"
benchmark_file = "test-human.py"
log_file = "description-{cot_mode}.log"

problem = ProblemReader(path, problem_description_file, benchmark_file, log_file.format(cot_mode=MODE_COT_HUMAN))

agent_params = AgentBehaviorParameters()
conversations = Conversations(problem=problem)
gpt_or = GPT4OR(agent_params=agent_params, conversations=conversations)

optimus = OptiMus(gpt_or=gpt_or)
optimus.solve_problem()
