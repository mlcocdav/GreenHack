import gradio as gr
import os
import openai

from backend import PromptHandler
from prompt_engineering import NLP_TASKS


def dummy_prompt(prompt, task_type, speed, quality):
    return f"{prompt} {task_type} {speed} {quality}", "text2", "text3", "text4", "text5"


def dummy_calculate_price(prompt, task_type, speed, quality):
    return (len(prompt) + speed + quality) * (NLP_TASKS.index(task_type) + 1)


if __name__ == "__main__":
    handler = PromptHandler()
    with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green,
                                           secondary_hue=gr.themes.colors.green)) as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    gr.Image(value="logo.png", show_label=False,
                             shape=[150, 30], interactive=False)
            with gr.Column():
                pass
            with gr.Column():
                pass
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt",
                                    placeholder="Enter your prompt here ...")
                task_type = gr.Dropdown(
                    label="Task type", choices=NLP_TASKS, value=NLP_TASKS[0],
                    interactive=True, allow_custom_value=False)
                speed = gr.Radio(list(range(1, 6)), label="Speed", value=3,
                                 interactive=True)
                quality = gr.Radio(list(range(1, 6)), label="Quality", value=3,
                                   interactive=True)

                with gr.Row():
                    simplified = gr.Checkbox(value=True, label="Simplify prompt", info="Simplify prompt", interactive=True)
                    out = gr.Textbox(label="Price per 1000 tokens - other providers*")

                prompt.change(handler.get_price_dollars, [prompt, task_type, speed, quality], out)
                task_type.change(handler.get_price_dollars, [prompt, task_type, speed, quality], out)
                speed.change(handler.get_price_dollars, [prompt, task_type, speed, quality], out)
                quality.change(handler.get_price_dollars, [prompt, task_type, speed, quality], out)

                try_btn = gr.Button("Try model")
            with gr.Column():
                model_output = gr.Textbox(label="Model response", lines=6)
                with gr.Row():
                    savings = gr.Textbox(label="SimpleGen price")
                    model_name = gr.Textbox(label="Selected model")
                    inference_speed = gr.Textbox(label="Inference time (ms)")

                shortened_prompt = gr.Textbox(label="Simplified Prompt", lines=5)
                    
                try_btn.click(
                    fn=handler.generate,
                    inputs=[prompt, task_type, speed, quality, simplified],
                    outputs=[model_output, inference_speed, model_name,
                             shortened_prompt, savings],
                    api_name="infer")
                with gr.Row():
                    like_btn = gr.Button("Wow, that's cool!",
                                         variant="primary")
                    dislike_btn = gr.Button("Nah, I want it better.")

        gr.Markdown(
            "\*  official price from other providers for the model with similar quality and speed")
    demo.launch()
