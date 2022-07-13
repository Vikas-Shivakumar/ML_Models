from flask import Flask, request, json, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
app=Flask(__name__)

Baggingclassifier = pickle.load(open("BaggingClassifier.pkl", "rb"))
RandomClassifier = pickle.load(open("RandomClassifier.pkl", "rb"))
CARTClassifier = pickle.load(open("CARTClassifier.pkl", "rb"))
XGBClassifier = pickle.load(open("XGBClassifier.pkl", "rb"))
AdaBoostclassifier = pickle.load(open("AdaBoostclassifier.pkl", "rb"))
KNNClassifier = pickle.load(open("KNNClassifier.pkl", "rb"))
NaiveBayesClassifier = pickle.load(open("NaiveBayesClassifier.pkl", "rb"))
SVCClassifier = pickle.load(open("SVCClassifier.pkl", "rb"))
LogisticClassifier = pickle.load(open("LogisticClassifier.pkl", "rb"))

tfidf = pickle.load(open("tfidf1.pkl", 'rb'))

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pandas as pd


def preprocess(messages):
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['Tweet'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

@app.route('/BaggingClassifier', methods=['POST'])
def BaggingClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =Baggingclassifier.predict(X_test)[0]
    
    return jsonify(str(pred))
 
@app.route('/RandomClassifier', methods=['POST']) 
def RandomClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =RandomClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))

@app.route('/CARTClassifier', methods=['POST']) 
def CARTClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =CARTClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))

@app.route('/XGBClassifier', methods=['POST']) 
def XGBClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =XGBClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))

@app.route('/AdaBoostclassifier', methods=['POST']) 
def AdaBoostclassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =AdaBoostclassifier.predict(X_test)[0]
    
    return jsonify(str(pred))

@app.route('/KNNClassifier', methods=['POST']) 
def KNNClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =KNNClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))

@app.route('/NaiveBayesClassifier', methods=['POST'])     
def NaiveBayesClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =NaiveBayesClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))    

@app.route('/SVCClassifier', methods=['POST']) 
def SVCClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =SVCClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))

@app.route('/LogisticClassifier', methods=['POST']) 
def LogisticClassifier_Predict():
    if request.method == 'POST': 
        text=request.json['text']
        data = {'Tweet':text}
      
        messages=pd.DataFrame(data, index=[0])
        text=preprocess(messages)
        # Create new tfidfVectorizer with old vocabulary
        tfidf_predict= TfidfVectorizer(max_features=2500,ngram_range=(1,3), vocabulary = tfidf.vocabulary_)
        X_test = tfidf_predict.fit_transform(text).toarray()

        pred =LogisticClassifier.predict(X_test)[0]
    
    return jsonify(str(pred))
   


    
if __name__ == '__main__':
    app.run(debug=False)