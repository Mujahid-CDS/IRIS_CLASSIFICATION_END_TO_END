import flask
from flask import request, render_template
import joblib

model = joblib.load('KNN_deployment_model.pkl')

app =  flask.Flask(__name__, static_url_path='')

@app.route('/', methods=['GET'])
def sendMainPage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def doPredict():
    sl=float(request.form['sl'])
    sw=float(request.form['sw'])
    pl=float(request.form['pl'])
    pw=float(request.form['pw'])
    X = [[sl,sw,pl,pw]]
    species = model.predict(X)[0]
    return render_template('predict.html', predict=species)
 
app.run()
