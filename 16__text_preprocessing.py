import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer 

text = "I am learning Natural Language Processing! NLP is used in chatbots, search engines and sentiment analysis. Visit https://nlp.com for more info."

# Step 1: Lowercase
text = text.lower()
print("Remove URLs:", text)

# Step 2: Remove URLs
text = re.sub(r'https?://\S+','',text)
print("Remove URLs:", text) 

# Step 3: Remove Punctuation
text = re.sub(r'[^a-zA-Z0-9\s]','',text)
print("Remove Punctuation:", text)

# Step 4: Tokenization 
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Step  5: Remove Stopwords 
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]
print("After Stopword Removal:", filtered) 

# Step 6: Stemming
stemmer = PorterStemmer()
stemmed  = [stemmer.stem(w) for w in filtered]
print("After Stemming:", stemmed) 

# Step 7: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
print("After Lemmatization:", lemmatized)