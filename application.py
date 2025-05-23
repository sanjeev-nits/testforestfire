import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from flask import Flask, render_template, request, redirect,jsonify

application = Flask(__name__)
app = application

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standed_scaler = pickle.load(open('models/scaler.pkl','rb'))

@app.route('/') 
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST','GET'])
def predict_datapoint():
    if request.method == "POST":
       temperature=float(request.form.get('Temperature'))
       RH=float(request.form.get('RH'))
       Ws=float(request.form.get('Ws'))
       Rain=float(request.form.get('Rain'))
       FFMC=float(request.form.get('FFMC'))
       DMC=float(request.form.get('DMC'))
       ISI=float(request.form.get('ISI'))
       Classes=float(request.form.get('Classes'))
       Region=float(request.form.get('Region'))
       print('the tempreture id',temperature)
       
       new_data_scaled=standed_scaler.transform([[temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
       result=ridge_model.predict(new_data_scaled)
       return render_template('/home.html',results=result[0])
    else:
        return render_template('/home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)