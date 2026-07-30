import gradio as gr
from transformers import pipeline


# Load YOUR trained model 
classifier = pipeline(
    "sentiment-analysis",
    model=".",
    tokenizer="."
)
def predict(text):
    if not text:
        return "Please enter some text"

    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "LABEL_1":
        return f"Positive (confidence: {score:.2f})"
    else:
        return f"Negative (confidence: {score:.2f})"


app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Enter text here...",
        label="Input Text"
    ),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Classifier",
    description="Enter any text and the model will predict if it is positive or negative"
)
app.launch()