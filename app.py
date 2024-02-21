from flask import Flask, render_template
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk

app = Flask(__name__)

# Load the data
food_restaurant = pd.read_csv('./data/grillogy.csv')



# Clean the data
food_restaurant.drop(['Tap5If', 'hCCjke', 'eaLgGf'], axis=1, inplace=True)
food_restaurant.dropna(inplace=True)
food_restaurant.rename(columns={'wiI7pd': 'reviews'}, inplace=True)

# Define preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
food_restaurant['cleaned_reviews'] = food_restaurant['reviews'].apply(preprocess_text)

# Import NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Define sentiment analysis functions
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Assuming tokenizer and model are defined elsewhere in your code
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Apply sentiment analysis
food_restaurant['sentiment'] = food_restaurant['cleaned_reviews'].apply(lambda x: sentiment_score(x[:300]))

# Split into positive and negative reviews
positive_reviews_df = food_restaurant[food_restaurant['sentiment'] >= 3]
negative_reviews_df = food_restaurant[food_restaurant['sentiment'] < 3]

# Define sentiment labels
labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/positive_reviews')
def positive_reviews():
    return render_template('positive_reviews.html', positive_reviews=positive_reviews_df)

@app.route('/negative_reviews')
def negative_reviews():
    return render_template('negative_reviews.html', negative_reviews=negative_reviews_df)

@app.route('/sentiment_distribution')
def sentiment_distribution():
    sentiment_counts = food_restaurant['sentiment'].value_counts().sort_index()
    return render_template('sentiment_distribution.html', sentiment_counts=sentiment_counts, labels=labels)

if __name__ == '__main__':
    app.run(debug=True)
