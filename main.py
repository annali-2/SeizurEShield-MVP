from flask import Flask, request, jsonify
import os
import tempfile
import torch
from preprocessing import preprocessing_functions

def predict(file, montage):
    # load model
    model = ResNet()
    model = torch.load(open('model.pckl', 'rb'))
    model.eval()

    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file provided'}), 400

    # file = request.files['file']
    # if file.filename == '':
    #     return jsonify({'error': 'No file selected'}), 400
    
    # # Get the 'montage' parameter from the request
    # montage = request.form.get('montage')
    if not montage:
        return jsonify({'error': 'No montage specified'}), 400

    # Run the prediction model
    try:
        # Preprocess the file using the specified montage
        preprocessed_data = preprocessing_functions.process(file, montage)
        
        # Example: Extract features and predict
        prediction = model.predict(preprocessed_data)  # Assuming model accepts preprocessed_data
        return prediction
        
    except Exception as e:
        # return jsonify({'error': str(e)}), 500
        print(e)

    

print('hi')
print(predict('aaaaaajy_s002_t000.edf', '01_tcp_ar'))
