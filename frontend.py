import gradio as gr
import os
import openai

TASK_TYPES = ["Chat", "Data Augmentation", "Summarization", "Text Generation", "Text Classification", "Question Answering", "Other"]

def openai_prompt(prompt, task_type, speed, quality):
	#print(os.getenv("OPENAI_API_KEY"))
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

def dummy_prompt(prompt, task_type, speed, quality):
	return f"{prompt} {task_type} {speed} {quality}", "text2", "text3", "text4", "text5"

def calculate_price(prompt, task_type, speed, quality):
	return (len(prompt) + speed + quality) * (TASK_TYPES.index(task_type) + 1)

if __name__ == "__main__":
	with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.green)) as demo:
		with gr.Row():
			with gr.Column():
				with gr.Row():
					gr.Image(value="logo.png", show_label=False, shape=[150,30], interactive=False)
			with gr.Column():
				pass
			with gr.Column():
				pass
		with gr.Row():
			with gr.Column():
				prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here ...")
				task_type = gr.Dropdown(label="Task Type", choices=TASK_TYPES, value=TASK_TYPES[0], interactive=True)
				speed = gr.Radio(list(range(1,6)), label="Speed", value=3, interactive=True)
				quality = gr.Radio(list(range(1,6)), label="Quality", value=3, interactive=True)

				out = gr.Number(label="Price per inference, $")

				prompt.change(calculate_price, [prompt, task_type, speed, quality], out)
				task_type.change(calculate_price, [prompt, task_type, speed, quality], out)
				speed.change(calculate_price, [prompt, task_type, speed, quality], out)
				quality.change(calculate_price, [prompt, task_type, speed, quality], out)

				try_btn = gr.Button("Try model")
			with gr.Column():
				model_output = gr.Textbox(label="Model response", lines=6)
				with gr.Row():
					inference_speed = gr.Textbox(label="Inference time, ms")
					model_name = gr.Textbox(label="Model Name")
					savings = gr.Textbox(label="*Savings, $")

				shortened_prompt = gr.Textbox(label="Improved Prompt", lines=4)
					
				try_btn.click(fn=dummy_prompt, inputs=[prompt, task_type, speed, quality], outputs=[model_output, inference_speed, model_name, shortened_prompt, savings], api_name="infer")
				with gr.Row():
					like_btn = gr.Button("Wow, that's cool!", variant="primary")
					dislike_btn = gr.Button("Nah, I wont it better")

		gr.Markdown("\* \- amount that can be saved if proposed model is used instead of ChatGPT")
	demo.launch()