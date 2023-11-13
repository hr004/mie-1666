from utils import get_templates
import os
from typing import Optional, Dict
api_keys = [
    os.getenv("OPENAI_API_KEY"),
    "sk-API-KEY-2",
    # ...
]

# Get templates
templates = get_templates()
template_formulation = templates["formulation"]
template_codegen = templates["codegen"]
template_codegen_var = templates["codegen_var"]
template_codegen_constr = templates["codegen_constr"]
template_codegen_objsolve = templates["codegen_objsolve"]
template_codefix_execution = templates["codefix_execution"]
template_codefix_data = templates["codefix_data"]
template_doublecheck = templates["doublecheck"]
template_rephrase = templates["rephrase"]
template_testgen = templates["testgen"]
template_standard_prompt = templates["standard_prompt"]

# Internal prompt
internal_prompt = (
    "You are an operations analyst and expert mathematical modeller AI bot."
)

# Behavior
MODE_COT_ONLY = 102
MODE_COT_DEBUG = 103
MODE_COT_DEBUG_TEST = 104
MODE_COT_HUMAN = 105

# Solution status
STATUS_PASSED = 0
STATUS_SYNTAX_ERROR = 1
STATUS_LOGIC_ERROR = 2



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
