import gradio as gr
import os
import openai
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
import ast

from gptrim import trim
from prompt_engineering import NLP_TASKS, NLP_TASK_PROMPTS, \
    NLP_TASK_TEMPERATURES

# Price for 1000 tokens
MODEL_PRICES = {
    "gpt-3.5-turbo": 0.002,
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.0005,
    "text-curie-001 ": 0.002,
    "text-davinci-003": 0.02,
}
# % of the price
PRICE_MARGIN = 0.1

# openai.api_key = os.environ['OPENAI_API']
openai.api_key = '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'


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


class PromptHandler():
    def __init__(self):
        self.prompt_history = pd.read_csv('prompt_history.csv')

    def generate(self, prompt: str, task, speed, quality, ):
        assert task in NLP_TASKS

        apis = APIs()
        df_subset = self.prompt_history[
            (self.prompt_history['speed'] == speed) &
            (self.prompt_history[
                 'quality'] == quality) &
            (self.prompt_history['task'] == task) &
            (self.prompt_history['feedback'])]
        embedding = apis.get_embedding(prompt)
        if df_subset.shape[0] > 0:
            df_subset.loc['prompt_embedding',:] = df_subset[
                'prompt_embedding'].apply(lambda x: ast.literal_eval(x))
            df_subset.loc['similarity',:] = df_subset.apply(
                lambda row: cos_sim(row['prompt_embedding'], embedding),
                axis=1)
            best_model = df_subset.sort_values(by=['similarity'])['model'].iloc[0]
        else:
            best_model = "text-ada-001"
        edited_prompt = self.simplify_prompt(
            f'{NLP_TASK_PROMPTS[task]}{prompt}')

        response_text = apis.openai_prompt(edited_prompt, best_model,
                                           temperature=NLP_TASK_TEMPERATURES[
                                               task])
        print('Response: ', response_text)
        # save prompt to db
        new_row = {
            'prompt': prompt, 'prompt_embedding': embedding,
            'model': best_model,
            'result': response_text, 'task': task,
            'speed': speed, 'quality': quality, 'feedback': True}
        self.prompt_history = pd.concat(
            [self.prompt_history, pd.DataFrame([new_row])], ignore_index=True)
        self.prompt_history.to_csv('prompt_history.csv', index_label='ID')

        return response_text

    def get_price(self, model: str) -> float:
        """Calculate price per 1000 tokens according to the inputs."""
        cost = MODEL_PRICES.get(model)
        return cost * (1 + PRICE_MARGIN)

    def get_saved_amount(self, final_price: float,
                         simplified_promt_ratio: float) -> float:
        chat_gpt_cost = MODEL_PRICES.get("gpt-3.5-turbo")
        return chat_gpt_cost - final_price * simplified_promt_ratio

    def simplify_prompt(self, promt: str) -> str:
        return trim(promt)


class APIs():

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0][
            'embedding']

    def openai_prompt(self, prompt, model, temperature):
        if model == "gpt-3.5-turbo":
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt}])
            return response['choices'][0]["message"]['content']
        else:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=100,
                top_p=1,
                n=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            return response['choices'][0]['text']


# iface = gr.Interface(fn=openai_prompt, inputs=["text"], outputs="text")
# iface.launch()


prompt = 'Hi, are you a human?'
ph = PromptHandler()
response = ph.generate(prompt, task='chat', speed=5, quality=4)
