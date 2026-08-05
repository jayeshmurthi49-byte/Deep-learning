import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv('Resume/Resume.csv')
print("Shape:", df.shape)
print("Categories:", df['Category'].nunique())
print(df['Category'].value_counts())

# Step 2: Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])
print("Classes:", list(le.classes_))

# Step 3: Train test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print("Train size:", len(train_df))
print("Test size:", len(test_df))

# Step 4: Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 5: Custom Dataset class
class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) 
                for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ResumeDataset(
    train_df['Resume_str'], 
    train_df['label']
)
test_dataset = ResumeDataset(
    test_df['Resume_str'], 
    test_df['label']
)

# Step 6: Load model
num_labels = df['label'].nunique()
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=num_labels
)

# Step 7: Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Step 8: Training arguments
training_args = TrainingArguments(
    output_dir="./resume_classifier",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True
)

# Step 9: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Step 10: Train
print("Starting training...")
trainer.train()

# Step 11: Save model
model.save_pretrained("./resume-classifier-model")
tokenizer.save_pretrained("./resume-classifier-model")

# Save label encoder classes
import json
with open('./resume-classifier-model/label_classes.json', 'w') as f:
    json.dump(list(le.classes_), f)

print("Model saved!")
print("Label classes saved!")

# Step 12: Test prediction
from transformers import pipeline
classifier = pipeline(
    "text-classification",
    model="./resume-classifier-model",
    tokenizer="./resume-classifier-model"
)

sample_resume = "Experienced Python developer with 3 years in machine learning and deep learning projects"
result = classifier(sample_resume)
print("\nTest prediction:", result)
predicted_label = int(result[0]['label'].split('_')[1])
print("Predicted category:", le.classes_[predicted_label])