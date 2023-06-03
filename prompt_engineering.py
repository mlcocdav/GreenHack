NLP_TASKS = ['Chat', 'Summarization', 'Classification', 'Question-answering', 'Generation', 'Other']

NLP_TASK_PROMPTS = {
	"Summarization": "summarize the following text ###",
	"Classification": "Classify the sentiment in the following text",
	"Question-answering": "Answer the question based on the context below. If the question cannot be answered using the information provided answer with 'I don't know'.",
	"Chat": "The below is a conversation with a funny chatbot. The chatbot's responses are amusing and entertaining.",
	"Generation": "Create a text using the following instructions",
	"Other":  "Answer the question based on the context below. If the question cannot be answered using the information provided answer with 'I don't know'.",
}

NLP_TASK_TEMPERATURES = {
	"Summarization": 0.0,
	"Chat": 0.2,
	"Classification": 0.0,
	"Question-answering": 0.2,
	"Generation": 1.0,
	"Other": 0.0
}
