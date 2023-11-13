from src.configure import AgentBehaviorParameters
from src.messages import Messages
from typing import Optional
from src.llms import CachedChatOpenAI
import os


class GptOr:
    def __init__(self, 
                 agent_params: AgentBehaviorParameters, 
                 conversations: Messages
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

    @property
    def automatic_test(self):
        if self.agent_params["Debug"] and not self.agent_params["Test"] and not self.agent_params["Human"]:
            return True
        return False
