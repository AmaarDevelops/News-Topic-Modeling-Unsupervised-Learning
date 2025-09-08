from flask import Flask,request,render_template,jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import string
from flask_cors import CORS


app = Flask(__name__)

CORS(app,origins=['http://localhost:500','http://127.0.0.1:5000'])

try:
    model = joblib.load('NMF_Model.joblib')
    vectorizer = joblib.load('Vectorizer.joblib')
    print('Model and vectorizer were loaded')
    predicted_ready = True
    feature_names = vectorizer.get_feature_names_out()
except Exception as e:
    print(f'Error occured while loadin the model :- {e}')
    vectorizer = None
    model = None
    feature_names = None
    prediction_ready = False


def lowercasing(txt):
    return txt.lower()

def remove_stopwords(txt):
    stop_words = set(stopwords.words('english'))
    try:
        words = word_tokenize(txt)
    except Exception as e:
        print(f'Error occured while removing stopwords :- {e}')
        return ""
    
    cleaned_words = [i.lower() for i in words if i.isalpha() and i.lower() not in stop_words]
    return " ".join(cleaned_words)

def remove_numbers(txt):
    return "".join([i for i in txt if not i.isdigit()])

def remove_punctuation(txt):
    return txt.translate(str.maketrans('','',string.punctuation))


def preprocess_txt(text):
    text = lowercasing(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = text.replace('\n','').replace('\r','')
    text = remove_stopwords(text)
    return text


topic_names = {
    0: "Business & Finance",
    1: "Politics",
    2: "Sports (Rugby & General)",
    3: "Film & Awards",
    4: "Economics & Growth",
    5: "Government & Law",
    6: "Olympics & Athletics",
    7: "Energy & Oil",
    8: "Technology & Software",
    9: "Mobile & Digital Media"
}

def get_top_words(topic_id,num_words=300):
    if not predicted_ready:
        return []
    
    topic_weights = model.components_[topic_id]
    top_word_indices = topic_weights.argsort()[:-num_words -1:-1]

    top_words = []
    for i in top_word_indices:
        word = feature_names[i]
        weight = float(topic_weights[i])
        top_words.append([word,weight])
    return top_words
    
    


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error' : 'Model not loaded. Please check server logs'})
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error' : 'No Text field provided, please provide it'})
    
    input_text = data['text']
    preprocessed_text = preprocess_txt(input_text)

    if not preprocessed_text.strip():
        return jsonify({
            'predicted_topic' : 'N/A',
            'topic_id' : -1,
            'confidence' : 0.00,
            'message' : 'Input text was filtered out during preprocessing. Try a longer, more descriptive text.'
        })

    vectorized_text = vectorizer.transform([preprocessed_text])
    topic_distribution = model.transform(vectorized_text)

    dominant_topic_id = topic_distribution.argmax()
    topic_distribution_probability = topic_distribution.max()

    predicted_topic_name = topic_names.get(dominant_topic_id,'Unknown Id')

    top_words = get_top_words(dominant_topic_id)

    response = {
        'predicted_topic' : predicted_topic_name,
        'topic_id' : int(dominant_topic_id),
        'confidence' : float(topic_distribution_probability),
        'top_words' : top_words
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)







