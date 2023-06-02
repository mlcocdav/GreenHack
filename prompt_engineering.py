NLP_TASKS = [ 'chat', 'summarization', 'classification', 'question-answering', 'generation', 'other']

NLP_TASK_PROMPTS = {
	"summarization": "summarize the following text ###",
	"classification": "Classify the sentiment in the following text",
	"question-answering": "Answer the question based on the context below. If the question cannot be answered using the information provided answer with 'I don't know'.",
	"chat": "The below is a conversation with a funny chatbot. The chatbot's responses are amusing and entertaining.",
	"generation": "Create a text using the following instructions",
	"other":  "Answer the question based on the context below. If the question cannot be answered using the information provided answer with 'I don't know'.",
}

NLP_TASK_TEMPERATURES = {
	"summarization": 0.0,
	"chat": 1.0,
	"classification": 0.0,
	"question-answering": 0.5,
	"generation": 1.0,
	"other": 0.0
}
