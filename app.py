import gradio as gr
import os
import openai
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
import ast

NLP_TASKS = ['summarization', 'classification', 'question-answering',
             'generation', 'chat', 'other']
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

openai.api_key = os.environ['OPENAI_API']


def cos_sim(a, b):
    a = ast.literal_eval(a)
    b = ast.literal_eval(b)
    return dot(a, b) / (norm(a) * norm(b))

class PromptHandler():
    def __init__(self):
        self.prompt_history = pd.read_csv(
            'prompt_history.csv')  # table with previous prompts
        self.model_repository = None  # dict with models?

    def generate(self, prompt: str, task, speed, quality, ):
        assert task in NLP_TASKS

        apis = APIs()
        #embedding = apis.get_embedding(prompt)
        df_subset = self.prompt_history[(self.prompt_history['speed'] == speed) &
                                        (self.prompt_history[
                                            'quality'] == quality) &
                                         (self.prompt_history['task'] == task) &
                                          (self.prompt_history['feedback'])]
        r = cos_sim(df_subset['prompt_embedding'].iloc[0],df_subset['prompt_embedding'].iloc[0])
        embedding = apis.get_embedding(prompt)
        if df_subset.shape[0] > 0:
            df_subset['similarity'] = df_subset.apply(lambda row: cos_sim(row['prompt_embedding'], embedding), axis=1)
            best_model = df_subset.groupby(['model']).mean('feedback').sort_values().iloc[0]

        # closest_prompts = self.find_closest_prompts(self.db, embedding)

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

    def openai_prompt(self, prompt, model):
        # print(os.getenv("OPENAI_API_KEY"))

        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=0,
            max_tokens=100,
            top_p=1,
            n=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return response['choices'][0]['text']


iface = gr.Interface(fn=openai_prompt, inputs=["text"], outputs="text")
iface.launch()

