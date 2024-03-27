from flask import Flask, render_template, request
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('ML_project(1).html')

@app.route('/submit', methods=['POST'])
def submit():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['chestPainType'])
    trestbps = float(request.form['restingBP'])
    chol = float(request.form['cholesterol'])
    fbs = int(request.form['fastingBloodSugar'])
    restecg = int(request.form['restECG'])
    thalach = float(request.form['maxHeartRate'])
    exang = int(request.form['exerciseAngina'])
    oldpeak = float(request.form['oldPeak'])
    slope = int(request.form['slope'])
    ca = float(request.form['numVessels'])
    thal = int(request.form['thal'])
    test = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    model = joblib.load('model1.joblib')
    res = model.predict(test)
    if res[0]==1:
        hel = "You are having heart disease"
    else:
        hel = 'No Disease Found'
    return render_template('ML_project(1).html',re=hel)

if __name__ == '__main__':
    app.run(debug=True)
