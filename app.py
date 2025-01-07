from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

import joblib
import re
import streamlit as st

def reg_text(text):
    text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip().lower()
    return text

wnl = WordNetLemmatizer()

def lemmatize_tokens(tokens):
        lemmatized_words = []
        #Loop through the tokens in each column and lemmatized the word based on verb
        for t in tokens:
            lemmatized_word = wnl.lemmatize(t, pos='v')
            lemmatized_words.append(lemmatized_word)
        return lemmatized_words

# install nessary files
nltk.download('punkt_tab')
nltk.download('')

st.title("Sentiment Analysis")

txt = st.text_area("Write Something...")

isAnalyse = st.button('Analyze')

# Text Classification Part
if isAnalyse:

    # Get Text
    cleaned_text = reg_text(txt)

    # Get Token
    tokenized = word_tokenize(cleaned_text)

    #Define appropriate stopwords based on the scenario, Spotify app review
    custom_stopwords = {'spotify', 'music', 'play', 'playlist', 'app', 'song', 'songs', 'podcast', 'get', 'let'}

    #Stopwords removal
    stopwords = nltk.corpus.stopwords.words("english") + list(custom_stopwords)
    filtered_token = [word for word in tokenized if word not in stopwords]

    #Lemmatization
    lemmatized_token = lemmatize_tokens(filtered_token)

    #Text vectorization using TF-IDF (Load In)
    tfidf_token = ' '.join(lemmatized_token)

    tf = joblib.load('vectorizer.pkl')
    tfidf_final = tf.transform([tfidf_token])

    # Load Model
    classification_model = joblib.load('svm.pkl')

    # Classify Sentiment
    predictions = classification_model.predict(tfidf_final)
    
    # Decode prediction
    if predictions == 1:
         st.write('Positive Sentiment!')
    else:
         st.write('Negative Sentiment!')
    



