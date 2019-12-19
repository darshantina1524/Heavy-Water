import flask
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask('Testing-Design') # Name of the App

@app.route('/') # Rendering to Test.html for root/index page
def home():
    return render_template('Test.html')

@app.route('/words', methods = ['POST']) # When the url /words is requested we would predict the label for new document
def predict():
    sentence = [i for i in request.form.values()] # Reads the data from the form

    # Loading the CountVectorizer and TfidfTransformer that we used to transform the training data
    count_Vec = pickle.load(open('Count_Vectorizer.pkl','rb'))
    tfidf = pickle.load(open('TfidfTransformer.pkl','rb'))

    # Loading the pre-trained model ( best model based on accuracy and recall on training and validation set)
    model = pickle.load(open('model.pkl','rb'))

    # Transforming the document entered by the user
    X = count_Vec.transform(sentence)
    X_data = tfidf.transform(X)
    # Predicting the label and probability of assignment for the new document
    y_pred = model.predict(X_data)
    prob = model.predict_proba(X_data)

    # Rendering result.html along with predicted label and and confidence of assignment
    return render_template("result.html", prediction = y_pred[0],probability=max(prob[0])*100)


if __name__ == '__main__':
    app.run(port = 5000, debug=True)
    # app.run(host = "0.0.0.0",port = 80, debug=True, threaded= True) - Used this command for AWS
