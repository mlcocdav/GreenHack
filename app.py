import gradio as gr
import os
import openai

def greet(name):
    return "Hello " + name + "!!"

def openai_prompt(prompt):
    #print(os.getenv("OPENAI_API_KEY"))
    openai.api_key = os.environ['OPENAI_API']

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
    return response['choices'][0]['text']

iface = gr.Interface(fn=openai_prompt, inputs="text", outputs="text")
iface.launch()