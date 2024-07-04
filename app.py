from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import tempfile
import pickle
import os
import mne

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'edf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/upload')
def upload_egg():
    return render_template('uploads.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ensure the uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Process the .edf file and generate output using MNE-Python
        output = process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('results.html', filename=filename, channels=output['channels'], duration=output['duration'])
    else:
        flash('File type not allowed')
        return redirect(request.url)
    
def process_file(filepath):
    # Add your .edf file processing logic using MNE-Python
    raw = mne.io.read_raw_edf(filepath)
    info = raw.info
    channels = info['ch_names']
    duration = raw.times[-1]
    return {'channels': channels, 'duration': duration}

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
