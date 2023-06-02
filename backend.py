import time
import timeit

import gradio as gr
import os
import openai
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np
import ast

from gptrim import trim
from prompt_engineering import NLP_TASKS, NLP_TASK_PROMPTS, NLP_TASK_TEMPERATURES

# Price for 1000 tokens
MODEL_PRICES = {
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.0005,
    "gpt-3.5-turbo": 0.002,
    "text-curie-001 ": 0.002,
    "text-davinci-003": 0.02,
}
# % of the price
PRICE_MARGIN = 0.1

openai.api_key = os.environ['OPENAI_API']

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
            # df_subset['prompt_embedding'] = df_subset['prompt_embedding'].apply(lambda x: ast.literal_eval(x))
            # df_subset['similarity'] = df_subset.apply(
            #     lambda row: cos_sim(row['prompt_embedding'], embedding),
            #     axis=1)
            # best_model = df_subset.sort_values(by=['similarity'])['model'].iloc[0]
            model_id = int(round((speed + quality) / 2, 0)) - 1
            best_model = list(MODEL_PRICES.keys())[model_id]
        else:
            best_model = "text-ada-001"
        simplified_prompt = self.simplify_prompt(prompt)
        edited_prompt = self.simplify_prompt(
            f'{NLP_TASK_PROMPTS[task]}{simplified_prompt}')

        # set up timer
        start_time = time.time()
        response_text = apis.openai_prompt(edited_prompt, best_model,
                                           temperature=NLP_TASK_TEMPERATURES[
                                               task])
        response_text = response_text.strip()
        inference_time = round(time.time() - start_time, 2)
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
        price = self.get_price(prompt, task, speed, quality)
        saved_money = self.get_saved_amount(price, len(edited_prompt)/len(simplified_prompt))

        return response_text, inference_time, best_model, simplified_prompt, saved_money

    def get_price(self, prompt: str, task_type: str, speed: int, quality: int) -> float:
        """Calculate price per inferences according to the inputs. 1 token is equal to 4 characters."""
        model_id = int(round((speed + quality)/2, 0)) - 1
        price_per_thousand = MODEL_PRICES[list(MODEL_PRICES.keys())[model_id]]
        return round(price_per_thousand * (1 + PRICE_MARGIN), 6)

    def get_saved_amount(self, final_price: float,
                         simplified_promt_ratio: float) -> float:
        chat_gpt_cost = MODEL_PRICES.get("gpt-3.5-turbo")
        return 100 * (1 - (final_price * simplified_promt_ratio)/ chat_gpt_cost)

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


# prompt = 'Hi, are you a human?'
# ph = PromptHandler()
# response = ph.generate(prompt, task='chat', speed=5, quality=4)
