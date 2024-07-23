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
    session
)
from werkzeug.utils import secure_filename

from preprocessing.preprocess import EEGMontage


# Configure Flask App 
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["DATA_FOLDER"] = "data/"
app.config["ALLOWED_EXTENSIONS"] = {"edf"}
app.secret_key = os.urandom(24)  # Set a unique and secret key

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
    rnn_model = RNNLSTM(input_size, hidden_size, output_size, num_layers)
    state_dict = torch.load(model_path)
    rnn_model.load_state_dict(state_dict)
    rnn_model.eval()
    return rnn_model


# Class for RNNLSTM model
class RNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out, _ = self.lstm(out, (h0, c0))
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

montages = ['01_tcp_ar', '02_tcp_le', '03_tcp_ar_a', '04_tcp_le_a', 'hello']

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
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
            session['uploaded_file'] = filename  # Store the filename in session
            # Process the .edf file and generate output using MNE-Python
            output = process_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return render_template(
                "results.html",
                montages=montages,
                filename=filename,
                channels=output["channels"],
                duration=output["duration"],
            )
        else:
            flash("File type not allowed")
            return redirect(request.url)
    else:  # GET request
        filename = session.get('uploaded_file')
        if filename:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.exists(filepath):
                output = process_file(filepath)
                return render_template(
                    "results.html",
                    montages=montages,
                    filename=filename,
                    channels=output["channels"],
                    duration=output["duration"],
                )
            else:
                flash("File not found")
                return redirect(url_for('upload_eeg'))
        else:
            flash("No file uploaded yet")
            return redirect(url_for('upload_eeg'))

# Run Model Prediction 
@app.route("/predict_file", methods=["GET", "POST"])
def predict_file():
    if request.method == "POST":
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

        selected_montage = request.form.get('montage')
        session['selected_montage'] = selected_montage

        # loop through the files but the assumption for now is that there is only one file
        for file in uploaded_files:
            # Get file from uploads and preprocess them
            eeg_montage = EEGMontage()
            edf_csv = eeg_montage.read_edf_to_dataframe(os.path.join("uploads", file))

            if selected_montage in eeg_montage.montage_dict:
                montage = eeg_montage.montage_dict[selected_montage]
            else:
                raise ValueError(f"Selected montage '{selected_montage}' is not available")
            edf_preprocessed = eeg_montage.compute_differential_signals(edf_csv, montage)
            edf_preprocessed2 = edf_preprocessed.fillna(0).drop(columns=["file_path"]).values

            # make predictions
            data = torch.stack([torch.tensor(d).float() for d in edf_preprocessed2])
            data = data.view(data.size(0), 1, input_size)
            prediction = model(data)
            probabilities = torch.softmax(prediction, dim=1)  # Softmax to get probabilities
            predicted_classes = torch.argmax(probabilities, dim=1)  # Predicted class indices (0 or 1)

            # add predictions to the preprocessed files
            predicted_classes_series = pd.Series(predicted_classes.numpy(), name="predicted_class")
            edf_preprocessed_with_classes = edf_preprocessed.assign(predicted_class=predicted_classes_series)
            predictions = edf_preprocessed_with_classes.head(10).to_dict(orient="records")

            filename_processed = f"{file.split('.')[0]}_processed.csv"
            processed_path = os.path.join(app.config["DATA_FOLDER"], filename_processed)
            edf_preprocessed_with_classes.to_csv(processed_path, index=False)

            additional_info.append({
                "file_name": file,
                "processed_file_name": f"{file.split('.')[0]}_processed.csv",
                "duration": 10.5,  # Example duration
            })

        session['predictions'] = predictions
        session['additional_info'] = additional_info
        return render_template("predictions.html", predictions=predictions, additional_info=additional_info)
    else:
        predictions = session.get('predictions')
        additional_info = session.get('additional_info')
        if predictions and additional_info:
            return render_template("predictions.html", predictions=predictions, additional_info=additional_info)
        else:
            flash("No predictions available. Please upload a file and run the prediction first.")
            return redirect(url_for('upload_eeg'))

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
        x=alt.X('timestamp', title='Timestamp (seconds)', axis=alt.Axis(titleFontSize=16, labelFontSize=14)),  # Adjust x-axis title and label font size
        y=alt.Y('Amplitude', title='Amplitude', axis=alt.Axis(titleFontSize=16, labelFontSize=14)),  # Adjust y-axis title and label font size
        color=alt.Color('Channel', scale=alt.Scale(domain=eeg_columns, range=[colors[ch] for ch in eeg_columns]), 
                        legend=alt.Legend(title="EEG Channels", titleFontSize=16, labelFontSize=14)),
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
    return render_template("eeg_chart.html", title="EEG Data Visualization", chart_json=chart_json, filename=filename)

