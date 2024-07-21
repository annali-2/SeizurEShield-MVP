import os
import tarfile
import mne

import torch
import torch.nn as nn
import pandas as pd
import altair as alt
import random
alt.data_transformers.disable_max_rows()

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from preprocessing.preprocess import EEGMontage


# Configure Flask App 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["DATA_FOLDER"] = "data/"
app.config["ALLOWED_EXTENSIONS"] = {"edf"}

# Safeguard against malicious file uplaods
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Extract tar.gz model
def extract_model(archive_path, extract_path):
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_path)
    return extract_path

# Load Model
def load_model(model_path, input_size, hidden_size, output_size, num_layers):
    rnn_model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
    state_dict = torch.load(model_path)
    rnn_model.load_state_dict(state_dict)
    rnn_model.eval()
    return rnn_model


# Class for RNN model
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Home Page Route
@app.route("/")
def index():
    return render_template("index.html")

# Upload File Page
@app.route("/upload")
def upload_egg():
    return render_template("uploads.html")

# Run Prediction 
@app.route("/predict")
def predict_eeg():
    return render_template("predict.html")

# Process File for EGG stats/description 
def process_file(filepath):
    # Add your .edf file processing logic using MNE-Python
    raw = mne.io.read_raw_edf(filepath)
    info = raw.info
    channels = info["ch_names"]
    duration = raw.times[-1]
    return {"channels": channels, "duration": duration}

# Upload EEG file 
@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Ensure the uploads directory exists
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        # Process the .edf file and generate output using MNE-Python
        output = process_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return render_template(
            "results.html",
            filename=filename,
            channels=output["channels"],
            duration=output["duration"],
        )
    else:
        flash("File type not allowed")
        return redirect(request.url)

# Run Model Prediction 
@app.route("/predict_file", methods=["POST"])
def predict_file():

    uploaded_files = os.listdir("uploads")
    additional_info = []

    # Extract the model archive
    archive_path = "model.tar.gz"
    extract_path = "./model_dir"
    extract_model(archive_path, extract_path)
    model_path = os.path.join(extract_path, "model.pth")

    # Define model parameters (these should match those used during training)
    input_size = 23
    hidden_size = 128
    output_size = 2
    num_layers = 2

    # Load the model
    model = load_model(model_path, input_size, hidden_size, output_size, num_layers)

    # loop through the files but the assumption for now is that there is only one file
    for file in uploaded_files:

        # Get file from uploads and preprocess them
        eeg_montage = EEGMontage()
        edf_csv = eeg_montage.read_edf_to_dataframe(os.path.join("uploads", file))
        montage = eeg_montage.montage_pairs_01_tcp_ar
        edf_preprocessed = eeg_montage.compute_differential_signals(edf_csv, montage)
        edf_preprocessed2 = (
            edf_preprocessed.fillna(0).drop(columns=["file_path"]).values
        )

        # make predictions
        data = torch.stack([torch.tensor(d).float() for d in edf_preprocessed2])
        data = data.view(data.size(0), 1, input_size)
        prediction = model(data)
        probabilities = torch.softmax(prediction, dim=1)  # Softmax to get probabilities
        predicted_classes = torch.argmax(
            probabilities, dim=1
        )  # Predicted class indices (0 or 1)

        # add predictions to the preprocessed files
        predicted_classes_series = pd.Series(
            predicted_classes.numpy(), name="predicted_class"
        )
        edf_preprocessed_with_classes = edf_preprocessed.assign(
            predicted_class=predicted_classes_series
        )
        predictions = edf_preprocessed_with_classes.head(10).to_dict(orient="records")

        filename_processed = f"{file.split('.')[0]}_processed.csv"
        processed_path = os.path.join(app.config["DATA_FOLDER"], filename_processed)
        edf_preprocessed_with_classes.to_csv(processed_path, index=False)

        additional_info.append(
            {
                "file_name": file,
                "processed_file_name": f"{file.split('.')[0]}_processed.csv",
                "duration": 10.5,  # Example duration
            }
        )

    return render_template(
        "predictions.html", predictions=predictions, additional_info=additional_info
    )

# Download Prediction CSV
@app.route("/download/<path:filename>", methods=["GET"])
def download_file(filename):
    data_folder = app.config["DATA_FOLDER"]
    processed_filename = f"{filename.split('.')[0]}_processed.csv"
    return send_from_directory(data_folder, processed_filename, as_attachment=True)


@app.route("/visual/<filename>")
def visual(filename):
    processed_filename = f"data/{filename}"
    data = pd.read_csv(processed_filename)
    df = pd.DataFrame(data)
    
    eeg_columns = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
                'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3',
                'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

    colors = {channel: "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for channel in eeg_columns}

    df_filtered = df[['timestamp'] + eeg_columns + ['predicted_class']]
    df_melted = df_filtered.melt(id_vars=['timestamp'], var_name='Channel', value_name='Amplitude')

    eeg_chart = alt.Chart(df_melted).mark_line().encode(
        x=alt.X('timestamp', title='Timestamp (seconds)'),  # Rename x-axis to 'Time'
        y='Amplitude',
        color=alt.Color('Channel', scale=alt.Scale(domain=eeg_columns, range=[colors[ch] for ch in eeg_columns])),
        tooltip=['timestamp', 'Channel', 'Amplitude']
    ).properties(
        width=1000,
        height=500,
    ).interactive()

    seizure_chart = alt.Chart(df_filtered[df_filtered['predicted_class'] == 1]).mark_rule(
        color='pink',
        opacity=0.075
    ).encode(
        x='timestamp',
        size=alt.value(2)
    )

    combined_chart = alt.layer(eeg_chart, seizure_chart).interactive()

    chart_json = combined_chart.to_json()  # Serialize the chart to JSON
    return render_template("eeg_chart.html", title="EEG Data Visualization", chart_json=chart_json)