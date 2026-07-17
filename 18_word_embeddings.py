from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab') 


# Step 1: Prepare corpus
corpus = [
    "I love natural language processing",
    "NLP is a field of artificial intelligence",
    "Word2Vec learns word embeddings from text",
    "King and queen are related words",
    "Machine learning and deep learning are popular",
    "Python is used for machine learning",
    "NLP models understand human language"
] 

# Step 2: Tokenize
tokensized = [word_tokenize(sentence.lower()) for sentence in corpus]
print("Tokenized corpus:", tokensized)

# Step 3: Train Word2Vec model
model = Word2Vec(
    sentences=tokensized,
    vector_size=50,   # size of word vector
    window=3,         # context window size
    min_count=1,      # minimum word frequency
    sg=0,             # 0=CBOW, 1=Skip-gram
    epochs=100
)

# Step 4: Get word vector
print("\n Vector for language:",model.wv['language']) 

# Step 5: Find similar words 
print("\nnWord similar to leanring :",model.wv.most_similar('learning',topn=3))
print("\n words similar to nlp",model.wv.most_similar('nlp',topn=3))

# Step 6: Word similarity score 
print("\nSimilarity between 'machine' and 'learning':", 
      model.wv.similarity('machine', 'learning')) 

# Step 7: Try Skip-gram 
model_sg = Word2Vec(sentences=tokensized,vector_size=50,window=3,min_count=1,sg=1,epochs=100)
print("\nSkip-gram vector for 'language':", model_sg.wv['language'])
