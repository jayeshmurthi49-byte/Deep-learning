import torch
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification

# Step 1: Load BERT tokenizer and model 

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


 
# Step 2: Tokenize input 
text = "I love natural language processing"
inputs = tokenizer(text,return_tensors="pt")
print(inputs)
print("Inputs IDS",inputs['input_ids'])
print("Token",tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))


with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
print("\n Outputs shape",last_hidden_state.shape) 

cls_embedding = last_hidden_state[:,0,:]
print("CLS token embedding shape:", cls_embedding.shape) 
print("CLS embedding (first 10 values):",cls_embedding[0][:10])

print("=== BERT for Classification ===")
classifier = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

with torch.no_grad():
    outputs = classifier(**inputs)


logits = outputs.logits
print("Logits",logits)

predicted_class = torch.argmax(logits,dim=1).item()
labels = {0:"NEGATIVE",1:"POSITIVE"}
print("Predicted:", labels[predicted_class])