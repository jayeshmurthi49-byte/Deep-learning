import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punk_tab')

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer

text = "I am learning Natural Language Processing. NLP is used in many applications like chatbots, search engines and sentiment analysis."

# Step  1: Word Tokenization
word_token = word_tokenize(text)
print("word Tokens:",word_token)

# Ste p 2: Sentence Tokenization
sent_token = sent_tokenize(text)
print("Sentence Tokens:",sent_tokenize)

stop_words = set(stopwords.words('english'))
filtered = [w for w in word_token if w.lower() not in stop_words]
print("After Stopword Removal:", filtered)


stemmer = PorterStemmer()
stemmed  = [stemmer.stem(w) for w in filtered]
print("After Stemming:", stemmed) 

lemmatizer = WordNetLemmatizer()
lemmatized  = [lemmatizer.lemmatize(w) for w in filtered]
print("After Lemmatization:", lemmatized)
