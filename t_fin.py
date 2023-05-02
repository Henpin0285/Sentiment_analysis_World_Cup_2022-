# Define Flask app: Create a Flask app and define routes for the home page and prediction.

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv(r'C:\Users\henry\OneDrive\Desktop\Project Capstone\Data\final_tweets_log.csv')

# Preprocess text data:
df2 = df.dropna(subset=['Tweets_cleaned'])
df2['Tweets_cleaned'].fillna(value='', inplace=True)
df2.dropna(subset=['Tweets_cleaned'], inplace=True)
df2['Tweets_cleaned'] = df2['Tweets_cleaned'].apply(lambda x: x.lower())

X_train,X_test,y_train,y_test=train_test_split(df2['Tweets_cleaned'],df2['Sentiment'],test_size=0.2,
                                               random_state=32)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

lm = LogisticRegression()
lm.fit(X_train, y_train)

# Define Flask app
app = Flask(__name__, template_folder= 'templates')

@app.route('/')
def home():
    return render_template('home_t.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    text = request.form.get('Tweets_cleaned')
    print(f"Text input: {text}")
    
    vectorized_text = vectorizer.transform([text])
    sentiment = lm.predict(vectorized_text)[0]
    return render_template('t_prediction.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
