import gradio as gr
import os
import openai


class PromptHandler():
    def __init__(self, ):
        self.db = None  # table with previous prompts
        self.model_repository = None  # dict with models?

    def generate(self, prompt: str, task, speed, quality, ):
        pass

    def select_model(self, ):
        pass

    def get_price(self, prompt: str, task: str, speed: int,
                  quality: int) -> float:
        """Calculate price according to the inputs."""


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
