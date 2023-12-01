import logging

# from swarms.models.openai_models import OpenAI
# from langchain.chat_models import ChatOpenAI as OpenAIChat
from openai.error import RateLimitError
from swarms.models import OpenAIChat

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import os
import time
from abc import ABC, abstractmethod

import openai
from openai import ChatCompletion


class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass


class OpenAILanguageModel(AbstractLanguageModel):
    """

    OpenAI Language Model


    Args:
        api_key (str): OpenAI API key
        strategy (str): Strategy for generating thoughts. Choose from 'cot' (Chain of Thoughts) or 'gpt' (GPT-3)
        evaluation_strategy (str): Strategy for evaluating thoughts. Choose from 'value' or 'vote'
        api_base (str): Base path for OpenAI API
        api_model (str): Model name for OpenAI API
        enable_ReAct_prompting (bool): Enable ReAct prompting

    Examples:
    >>> from tree_of_thoughts.models.openai_models import OpenAILanguageModel
    >>> model = OpenAILanguageModel(api_key=api_key)
    >>> model.generate_thoughts(state, k)
    >>> model.evaluate_states(states, initial_prompt)

    """

    def __init__(
        self,
        api_key,
        strategy="cot",
        evaluation_strategy="value",
        enable_ReAct_prompting=True,
        *args,
        **kwargs,
    ):
        self.api_key = api_key
        self.use_chat_api = True
        self.enable_ReAct_prompting = enable_ReAct_prompting
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

        self.client = ChatCompletion(cache=True, api_key=os.getenv("OPENAI_API_KEY"))
        self.api_model = "gpt-4"
        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ""
        if enable_ReAct_prompting:
            self.ReAct_prompt = (
                "Write down your observations in format 'Observation:xxxx',"
                " then write down your thoughts in format 'Thoughts:xxxx'."
            )

        self.model = OpenAIChat(
            openai_api_key=api_key, model_name="gpt-4", *args, **kwargs
        )

    def generate_text(self, prompt: str, k: int = 3):
        """Generate text from prompt using OpenAI API"""
        if self.use_chat_api:
            thoughts = []
            for _ in range(k):
                response = self.model(prompt)
                thoughts += [response]
                # print(f'thoughts: {thoughts}')
            return thoughts

    def generate_thoughts(self, state, k, initial_prompt, rejected_solutions=None):
        """
        Generate thoughts from state using OpenAI API

        Args:
            state (str or list): State of reasoning
            k (int): Number of thoughts to generate
            initial_prompt (str): Initial prompt
            rejected_solutions (list): List of rejected solutions

        Returns:
            list: List of thoughts



        """
        if type(state) == str:
            state_text = state
        else:
            state_text = "\n".join(state)
        print("New state generating thought:", state, "\n\n")
        prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
        Considering the reasoning provided:\n\n
        ###'{state_text}'\n\n###
        Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
        ###{rejected_solutions}###, 
        complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them."""

        prompt += self.ReAct_prompt
        thoughts = self.generate_text(prompt, k)
        return thoughts

    def generate_solution(self, initial_prompt, state, rejected_solutions=None):
        try:
            if isinstance(state, list):
                state_text = "\n".join(state)
            else:
                state_text = state

            prompt = f"""You're an TreeofThoughts, an superintelligent AI model devoted to helping Humans by any means necessary. You're purpose is to generate a series of solutions to comply with the user's instructions, you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Wrap all the generated code using markdown triple backticks (```) to maintain format.
            Example:
            ```python
               import json
            ```
            Only provide the correct python code that you think will fix error.
            """
            answer = self.generate_text(prompt, 1)
            print(f"Answerrrrrr {answer}")
            # print(thoughts)
            # print(f"General Solution : {answer}")
            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == "value":
            state_values = {}
            for state in states:
                if type(state) == str:
                    state_text = state
                else:
                    state_text = "\n".join(state)
                prompt = f""" To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not directly concretely making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """

                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == "vote":
            states_text = "\n".join([" ".join(state) for state in states])
            prompt = (
                "Given the following states of reasoning, vote for the best"
                " state utilizing an scalar value"
                f" 1-10:\n{states_text}\n\nVote, on the probability of this"
                f" state of reasoning achieveing {initial_prompt} and become"
                " very pessimistic very NOTHING ELSE"
            )
            response = self.openai_api_call_handler(prompt, 50, 1)
            print(f"state response: {response}")
            best_state_text = self.openai_choice2text_handler(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            print(f"best_state: {best_state}")

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

    def openai_api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        while True:
            try:
                if self.use_chat_api:
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.create(
                        model=self.api_model,
                        messages=messages,
                        max_tokens=400,
                        temperature=temperature,
                    )
                else:
                    print("HUUU")
                    # response = openai.Completion.create(
                    #     engine="gpt-4",
                    #     prompt=prompt,
                    #     n=k,
                    #     max_tokens=max_tokens,
                    #     stop=stop,
                    #     temperature=temperature,
                    # )
                with open("openai.logs", "a") as log_file:
                    log_file.write(
                        "\n" + "-----------" + "\n" + "Prompt : " + prompt + "\n"
                    )
                return response
            except RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(
                    f"{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT"
                )
                time.sleep(sleep_duratoin)

    def openai_choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice["message"]["content"]
        else:
            text = choice.text.strip()
        return text
