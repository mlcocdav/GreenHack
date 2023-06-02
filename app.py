import gradio as gr
import os
import nltk
import openai

# Price for 1000 tokens"
MODEL_PRICES = {
    "gpt-3.5-turbo": 0.002,
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.00005,
    "text-curie-001 ": 0.002,
    "text-davinci-003": 0.02,
}

class PromptHandler():
    def __init__(self, ):
        self.db = None  # table with previous prompts
        self.model_repository = None  # dict with models?

    def generate(self, prompt: str, task, speed, quality, ):
        pass

    def select_model(self, ):
        pass

    def get_price(self, prompt: str, model: str) -> float:
        """Calculate price according to the inputs."""
        tokens_num = len(nltk.word_tokenize(prompt))
        return MODEL_PRICES.get(model) * tokens_num / 10000


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
