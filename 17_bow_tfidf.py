from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "I love natural language processing",
    "NLP is a field of artificial intelligence",
    "I love machine learning and deep learning",
    "Deep learning is part of machine learning"
] 

# Step 1: Bag of Words 
print("=== Bag of Words ===")
cv = CountVectorizer() 
bow_matrix = cv.fit_transform(corpus)
print("Vocabulary:", cv.get_feature_names_out())
print("BOw Matrix:\n", bow_matrix.toarray())

# Step 2: Bag of Words with Bigrams 
print("\n === Bigrams === ")
cv_bigram = CountVectorizer(ngram_range=(2,2))
bigram_matrix = cv_bigram.fit_transform(corpus)
print("Bigram Vocabulary:",cv_bigram.get_feature_names_out())

# Step 3: TF-IDF 
print("\n=== TF-IDF ===") 
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(corpus)
print("Vocabulary:", tfidf.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray().round(2))

# Step 4: Compare BOW vs TF-IDF for one sentence
print("\n=== BOW vs TF-IDF for sentence 1 ===") 
print("BOW:",bow_matrix.toarray()[0])
print("TF-IDF",tfidf_matrix.toarray()[0].round(2))

