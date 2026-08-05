import gradio as gr
import torch 
import json
from transformers import AutoTokenizer,AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./resume-classifier-model")
model = AutoModelForSequenceClassification.from_pretrained("./resume-classifier-model")
model.eval() 


# Load label classes
with open('./resume-classifier-model/label_classes.json', 'r') as f:
    label_classes = json.load(f)


def predict(resume_text):
    if not resume_text:
        return "please enter resume text"

    inputs = tokenizer(
        resume_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits,dim=1).item()

    return f"Predicted Job Category: {label_classes[predicted_class]}"

app = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Paste resume text here...",
        label="Resum Text",
        lines=10
    ),
    outputs=gr.Textbox(label="Predicted Category"),
    title="Resume Classifier",
    description="Paste resume text and the model will predict the job category"

)
app.launch()
