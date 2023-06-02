import gradio as gr
import os
import openai
import pandas as pd

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

    def get_price(self, prompt: str, task: str, speed: int,
                  quality: int) -> float:
        """Calculate price according to the inputs."""

class APIs():
    def __init__(self):
        pass

    def get_embedding(self, text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)['data'][0][
            'embedding']

def openai_prompt(prompt):
    # print(os.getenv("OPENAI_API_KEY"))
    openai.api_key = os.environ['OPENAI_API']

    response = openai.Completion.create(
        model="text-ada-001",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response['choices'][0]['text']


iface = gr.Interface(fn=openai_prompt, inputs=["text"], outputs="text")
iface.launch()
