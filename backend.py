import ast
import time

import nltk
import openai
import pandas as pd
from gptrim import trim
from numpy import dot
from numpy.linalg import norm
import os
from prompt_engineering import (NLP_TASKS, NLP_TASK_PROMPTS,
                                NLP_TASK_TEMPERATURES)

# Price for 1000 tokens
MODEL_PRICES = {
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.0005,
    "gpt-3.5-turbo": 0.002,
    "text-curie-001": 0.002,
    "text-davinci-003": 0.02,
}
# % of the price
PRICE_MARGIN = 0.0

openai.api_key = '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' # os.getenv("OPENAI_API_KEY")


def cos_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))


class PromptHandler():
    def __init__(self):
        self.prompt_history = pd.read_csv('prompt_history.csv')

    def generate(self, prompt: str, task, speed, quality, simplified):
        assert task in NLP_TASKS

        apis = APIs()
        df_subset = self.prompt_history[
            (self.prompt_history['speed'] == speed) &
            (self.prompt_history[
                 'quality'] == quality) &
            (self.prompt_history['task'] == task) &
            (~self.prompt_history['feedback'])]
        embedding = apis.get_embedding(prompt)
        if df_subset.shape[0] > 0:
            df_subset['prompt_embedding'] = df_subset[
                'prompt_embedding'].apply(lambda x: ast.literal_eval(x))
            df_subset['similarity'] = df_subset.apply(
                lambda row: cos_sim(row['prompt_embedding'], embedding),
                axis=1)
            best_model = \
            df_subset.sort_values(by=['similarity'])['model'].iloc[0]
        else:
            model_id = max(int(round((speed + quality) / 2, 0)) - 2, 0)
            print(model_id)
            best_model = list(MODEL_PRICES.keys())[model_id]

        if simplified:
            simplified_prompt = self.simplify_prompt(prompt)
            edited_prompt = self.simplify_prompt(
                f'{NLP_TASK_PROMPTS[task]}{simplified_prompt}')
            try:
                simplified_prompt_ratio = len(
                nltk.word_tokenize(simplified_prompt)) / len(
                nltk.word_tokenize(prompt))
            except:
                simplified_prompt_ratio = 1
        else:
            edited_prompt = f'{NLP_TASK_PROMPTS[task]}{prompt}'
            simplified_prompt_ratio = 1
            simplified_prompt = ''

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
        price = round(self.get_price(prompt, task, speed, quality,
                               model_name=best_model), 6)
        competitor_price = self.get_price(prompt, task, speed, quality)
        savings_percent = round(
            self.get_saved_amount(price, competitor_price,
                                  simplified_prompt_ratio), 2)
        saved_money = f'${round(price * simplified_prompt_ratio, 6)} saved {savings_percent} % '
        return response_text, inference_time, best_model, simplified_prompt, saved_money

    def get_price(self, prompt: str, task_type: str, speed: int, quality: int,
                  model_name=None) -> float:
        """Calculate price per inferences according to the inputs."""
        if model_name is None:
            model_id = int(round((speed + quality) / 2, 0)) - 1
            price_per_thousand = MODEL_PRICES[
                list(MODEL_PRICES.keys())[model_id]]
        else:
            price_per_thousand = MODEL_PRICES[model_name]
        return round(price_per_thousand * (1 + PRICE_MARGIN), 6)

    def get_price_dollars(self, prompt: str, task_type: str, speed: int,
                          quality: int,
                          model_name=None) -> float:
        """Calculate price per inferences according to the inputs."""
        if model_name is None:
            model_id = int(round((speed + quality) / 2, 0)) - 1
            model_name = list(MODEL_PRICES.keys())[model_id]
        return f'${self.get_price(prompt, task_type, speed, quality, model_name)} ({model_name})'

    def get_saved_amount(self, final_price: float, other_price: float,
                         simplified_promt_ratio: float) -> float:
        return 100 * (1 - (final_price * simplified_promt_ratio) / other_price)

    def simplify_prompt(self, promt: str) -> str:
        return trim(promt)


class APIs():

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        if text == '':
            text = ' '
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
