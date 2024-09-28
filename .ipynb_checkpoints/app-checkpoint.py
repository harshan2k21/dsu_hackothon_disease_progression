from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np

class HealthcarePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(HealthcarePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

app = Flask(__name__)

# Load the model and preprocessor
input_size = 10  # Replace with the actual number of features after preprocessing
model = HealthcarePredictionModel(input_size)
model.load_state_dict(torch.load('global_model.pth', weights_only=True))

model.eval()

preprocessor = joblib.load('preprocessor.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Preprocess the input data
    input_data = preprocessor.transform(pd.DataFrame([data]))
    input_tensor = torch.FloatTensor(input_data)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    
    return jsonify({'prediction': int(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True)

    from flask import render_template

@app.route('/')
def home():
    return render_template('index.html')