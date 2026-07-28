import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Step 0: Check GPU
print("CUDA available:", torch.cuda.is_available())

# Step 1: Load dataset
# Step 1: Load dataset
print("Loading dataset...")
dataset = load_dataset("stanfordnlp/imdb")
print(dataset)
print("Sample", dataset['train'][0])

# Step 2: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 3: Tokenize dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

tokenized = dataset.map(tokenize, batched=True, batch_size=512)
tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 4: Use small subset for faster training
train_dataset = tokenized['train'].shuffle(seed=42).select(range(2000))
eval_dataset = tokenized['test'].shuffle(seed=42).select(range(500))

print("Train size:", len(train_dataset))
print("Eval size:", len(eval_dataset))

# Step 5: Load pretrained BERT for classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Step 6: Training arguments (FIXED: eval_strategy, not evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./bert_sentiment",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Step 7: Train
print("Starting fine tuning...")
trainer.train()

# Step 8: Save model
model.save_pretrained("./bert-sentiment-model")
tokenizer.save_pretrained("./bert-sentiment-model")
print("Model saved!")

# Step 9: Test with pipeline
from transformers import pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="./bert-sentiment-model",
    tokenizer="./bert-sentiment-model"
)

print("\nTest predictions:")
print(classifier("This movie was absolutely amazing!"))
print(classifier("This was the worst movie I have ever seen."))