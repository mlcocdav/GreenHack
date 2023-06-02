import gradio as gr
import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def greet(name):
    return "Hello " + name + "!!"

def openai(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response

iface = gr.Interface(fn=openai, inputs="text", outputs="text")
iface.launch()