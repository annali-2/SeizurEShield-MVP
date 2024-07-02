from flask import Flask, request, jsonify
import os
import tempfile
import pickle
import mne

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # load model
    model = pickle.load(open('model.pckl', 'rb'))

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get the 'montage' parameter from the request
    montage = request.form.get('montage')
    if not montage:
        return jsonify({'error': 'No montage specified'}), 400

    # Run the prediction model
    try:
        # Preprocess the file using the specified montage
        preprocessed_data = preprocess_file(file_path, montage)
        
        # Example: Extract features and predict
        prediction = model.predict(preprocessed_data)  # Assuming model accepts preprocessed_data
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'prediction': prediction})

def preprocess_file(file_path, montage):
    # Load the EDF file using MNE
    raw = mne.io.read_raw_edf(file_path, preload=True)

    # Apply montage
    montage = mne.channels.read_montage(montage)
    raw.set_montage(montage)

    # Example preprocessing steps:
    # 1. Apply a bandpass filter
    raw.filter(l_freq=0.5, h_freq=50)

    # 2. Extract epochs (optional)
    events = mne.find_events(raw)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=1, baseline=None, preload=True)

    # Return preprocessed data
    return epochs.get_data()
