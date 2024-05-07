from flask import Flask, request, jsonify, render_template

import utility
import pickle
from model import ANNModel

app = Flask(__name__)
model = ANNModel()

@app.route('/')
def index():
    return "Welcome to the GhostPostCC Inference API!"

@app.route('/inference', methods=['POST'])
def inference_endpoint():
    data = request.json
    fuel_sources = data['fuel_sources']
    inference_timestamp = data['inference_timestamp']

    result = model.inference(fuel_sources, inference_timestamp)
    return jsonify(result)

@app.route('/train', methods=['POST'])
def train_endpoint():
    data = request.json
    fuel_sources = data['fuel_sources']

    result = model.forecast_all_fuel_sources(fuel_sources)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=8000, debug=True)