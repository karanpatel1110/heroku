import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import flasgger
from flasgger import Swagger
# create flask api
app1 = Flask(__name__)
#Swagger(app)

# load the pickle model
classifier = pickle.load(open("model_fish.pkl", "rb"))

@app1.route('/')
def home():
    return render_template('index.html')


@app1.route('/predicted', methods=["POST"])
def predict():
    fet = [x for x in request.form.values()]
    fet_fin = [np.array(fet)]
    pred = classifier.predict(fet_fin)
    return render_template('index.html',pred_text = 'the fish species is {}'.format(str(pred)))

if __name__ == "__main__":
    app1.run()
