# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('../tuned_pipeline.pkl','rb'))

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict_proba(pd.read_json(data))
    output = pd.DataFrame(prediction, columns= model.classes_.tolist())

    return output.to_json()


if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
    	print("Server is exited unexpectedly. Please contact server admin.")