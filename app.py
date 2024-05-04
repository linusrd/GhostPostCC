from flask import Flask, request, jsonify, render_template

import utility
import pickle
from model import ANNModel

app = Flask(__name__)
mode = ANNModel()

@app.route('/')
def index():
    return "Welcome to the GhostPostCC Inference API!"

@app.route('/inference', methods=['POST'])
def inference_endpoint():
    data = request.json
    fuel_sources = data['fuel_sources']
    inference_timestamp = data['inference_timestamp']

    result = mode.inference(fuel_sources, inference_timestamp)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=8000, debug=True)