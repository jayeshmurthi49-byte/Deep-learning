from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Step 1: Short text summarization
text1 = """
Natural language processing is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers 
and human language, in particular how to program computers to process and 
analyze large amounts of natural language data. The goal is a computer capable 
of understanding the contents of documents, including the contextual nuances 
of the language within them.
"""

summary1 = summarizer(text1, max_length=50, min_length=20, do_sample=False)
print("=== Summary 1 ===")
print("Original length:", len(text1.split()))
print("Summary:", summary1[0]['summary_text'])
print("Summary length:", len(summary1[0]['summary_text'].split()))

# Step 2: Longer text summarization
text2 = """
Machine learning is a method of data analysis that automates analytical model building. 
It is based on the idea that systems can learn from data, identify patterns and make 
decisions with minimal human intervention. Machine learning is a type of artificial 
intelligence that allows software applications to become more accurate at predicting 
outcomes without being explicitly programmed to do so. Machine learning algorithms 
use historical data as input to predict new output values. Recommendation engines are 
a common use case for machine learning. Other popular uses include fraud detection, 
spam filtering, malware threat detection, business process automation and predictive 
maintenance. Machine learning has become increasingly important in recent years as 
organizations seek to use data to make better decisions.
"""

summary2 = summarizer(text2, max_length=60, min_length=25, do_sample=False)
print("\n=== Summary 2 ===")
print("Original length:", len(text2.split()))
print("Summary:", summary2[0]['summary_text'])
print("Summary length:", len(summary2[0]['summary_text'].split()))

# Step 3: Summarize multiple texts at once
texts = [text1, text2]
summaries = summarizer(texts, max_length=50, min_length=20, do_sample=False)
print("\n=== Multiple Summaries ===")
for i, s in enumerate(summaries):
    print(f"Summary {i+1}:", s['summary_text'])