from transformers import pipeline,AutoTokenizer 

# Step 1: Sentiment Analysis Pipeline
classifier = pipeline("sentiment-analysis")
results = classifier("I love learning NLP with HuggingFace!")
print(results) 


result2 = classifier("This is the worst experience ever.")
print(result2)

# Step 2: Multiple sentences at once
print("\n=== Multiple Sentences ===")
sentences = [
    "HuggingFace makes NLP so easy",
    "I hate when things don't work",
    "The weather is okay today"
]
results  = classifier(sentences)
for sentence, result in zip(sentences, results):
    print(f"'{sentence}' → {result['label']} ({result['score']:.2f})")


# Step 3: Text Generation Pipeline
print("\n=== Text Generation ===")
generator = pipeline("text-generation",model="gpt2")
output = generator("Artificial Intelligence is", max_length=50, num_return_sequences=1)
print(output[0]['generated_text'])