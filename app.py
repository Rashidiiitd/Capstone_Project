from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from collections import Counter
from itertools import product

app = Flask(__name__)

# Load your machine learning model
model = pickle.load(open('rf_model_window_size_10.pkl', 'rb'))

# Define the function to compute DPC features
def compute_dpc(sequence):
    sequence = sequence.replace('^', '')  # Remove any special characters if present
    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
    dipeptides = [''.join(pair) for pair in product(amino_acids, repeat=2)]
    dpc_counts = Counter([sequence[i:i+2] for i in range(len(sequence)-1)])
    dpc_vector = [dpc_counts[dipeptide] for dipeptide in dipeptides]
    return dpc_vector

# Define the function to preprocess the sequence
def preprocess_sequence(sequence):
    sequence = sequence.strip()  # Remove spaces from front and end
    if len(sequence) != 17:
        return None, "Sequence length must be exactly 17 characters."
    sequence = sequence.upper()  # Convert to uppercase
    return sequence, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sequence = request.form['sequence']
    sequence, error = preprocess_sequence(sequence)
    
    if error:
        return jsonify({'error': error})
    
    dpc_vector = compute_dpc(sequence)
    dpc_vector = np.array(dpc_vector).reshape(1, -1)
    
    prediction = model.predict(dpc_vector)
    
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
