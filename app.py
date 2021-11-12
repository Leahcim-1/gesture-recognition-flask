from flask import Flask, request, Response
from joblib import load
import numpy as np
import pandas as pd
import os
import json
import sklearn
from threading import Thread

model_path = os.path.join(os.getcwd(), 'test.joblib')
clf = load(model_path)

app = Flask(__name__)


@app.route("/")
def index():
    return "<p>Hello, World!</p>"


def prediction(data):
    data = list(map(lambda acc: [acc['x'], acc['y'], acc['z']], data))
    data_set = [item for sublist in data for item in sublist]
    df = pd.DataFrame({'data': data_set})
    X = pd.DataFrame(df['data'].tolist(), index=df.index)
    print(X)
    pred = clf.predict(X)
    print(pred)


@app.route("/api/acc/predict", methods=['POST'])
def predict():
    data = request.json['data_set']
    Thread(target=lambda: prediction(data)).start()
    resp = Response(b'OK', status=200)
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=11000)
