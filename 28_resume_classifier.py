import pandas as pd 
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score 

# Step 1: Load dataset
df = pd.read_csv("Resume/Resume.csv")
print("Shape",df.shape)
print("Categories:",df['Category'].nunique()) 
print(df["Category"].value_counts())  

# Step 2: Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])
print('Classes:', list(le.classes_)) 

# Step 3: Train test split
train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)
print("Train size:", len(train_df))
print("Test size:", len(test_df)) 

# Step 4: Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Step 5: Custom Dataset class 
class ResumeDataset(Dataset):
    def __init__(self,texts,labels):
        self.encoding = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=128)
        
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k:torch.tensor(v[idx])
                for k,v in self.encoding.items()}
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