import gradio as gr
import os
import openai
import pandas as pd
from gptrim import trim

# Price for 1000 tokens
MODEL_PRICES = {
    "gpt-3.5-turbo": 0.002,
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.00005,
    "text-curie-001 ": 0.002,
    "text-davinci-003": 0.02,
}
# % of the price
PRICE_MARGIN = 0.1

NLP_TASKS = ['summarization', 'classification', 'question-answering', 'generation', 'chat', 'other']

class PromptHandler():
    def __init__(self, ):
        self.db = pd.read_csv('prompt_history.csv')  # table with previous prompts
        self.model_repository = None  # dict with models?

    def generate(self, prompt: str, task, speed, quality, ):
        assert task in NLP_TASKS

        # shorten prompt?
        # embed prompt, find previous prompts with similar embeddings, select model that worked best for them
        apis = APIs()
        embedding = apis.get_embedding(prompt)
        # find similar prompts in db

    def select_model(self, ):
        pass

    def get_price(self, model: str) -> float:
        """Calculate price per 1000 tokens according to the inputs."""
        cost = MODEL_PRICES.get(model) 
        return cost * (1 + PRICE_MARGIN)
    def get_saved_amount(self, final_price: float, simplified_promt_ratio: float) -> float:
        chat_gpt_cost = MODEL_PRICES.get("gpt-3.5-turbo")
        return chat_gpt_cost - final_price * simplified_promt_ratio

    def simplify_prompt(self, promt: str) -> str:
        return trim(promt)
    
class APIs():
    def __init__(self):
        pass

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0][
            'embedding']

# def openai_prompt(prompt):
#     # print(os.getenv("OPENAI_API_KEY"))
#     openai.api_key = os.environ['OPENAI_API']
#
#     response = openai.Completion.create(
#         model="text-ada-001",
#         prompt=prompt,
#         temperature=0,
#         max_tokens=100,
#         top_p=1,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#         stop=["\n"]
#     )
#     return response['choices'][0]['text']
#
#
# iface = gr.Interface(fn=openai_prompt, inputs=["text"], outputs="text")
# iface.launch()
