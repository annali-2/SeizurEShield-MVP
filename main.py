import tarfile
import os
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd

#from preprocessing import preprocessing_functions
from preprocessing.preprocess import EEGMontage


def extract_model(archive_path, extract_path):
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    return extract_path

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
    
def load_model(model_path, input_size, hidden_size, output_size, num_layers):
    rnn_model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
    state_dict = torch.load(model_path)
    rnn_model.load_state_dict(state_dict)
    rnn_model.eval()
    return rnn_model


if __name__ == '__main__':
    # Step 1: Extract the model archive
    archive_path = 'model.tar.gz'
    extract_path = './model_dir'
    extract_model(archive_path, extract_path)

    model_path = os.path.join(extract_path, 'model.pth')

    # Define your model parameters (these should match those used during training)
    input_size = 23  # Example input size
    hidden_size = 128  # Example hidden size
    output_size = 2  # Example number of output classes
    num_layers = 2  # Example number of RNN layers

    model = load_model(model_path, input_size, hidden_size, output_size, num_layers)
    print('Model loaded successfully!')
    print(model)

    eeg_montage = EEGMontage()
    edf_csv = eeg_montage.read_edf_to_dataframe('aaaaaajy_s002_t000.edf')
    montage = eeg_montage.montage_pairs_01_tcp_ar
    edf_preprocessed = eeg_montage.compute_differential_signals(edf_csv, montage)
    edf_preprocessed2 = edf_preprocessed.fillna(0).drop(columns=['file_path']).values

    # with torch.no_grad():
    data = torch.stack([torch.tensor(d).float() for d in edf_preprocessed2])
    data = data.view(data.size(0), 1, input_size)
    prediction = model(data)
    probabilities = torch.softmax(prediction, dim=1)  # Softmax to get probabilities
    predicted_classes = torch.argmax(probabilities, dim=1)  # Predicted class indices (0 or 1)

    print('Prediction:', prediction)
    print('Probabilities:', probabilities)
    print('Predicted classes:', predicted_classes)



    # Convert predicted_classes tensor to a Pandas Series and add predicted_classes as a column to edf_preprocessed
    predicted_classes_series = pd.Series(predicted_classes.numpy(), name='predicted_class')
    edf_preprocessed_with_classes = edf_preprocessed.assign(predicted_class=predicted_classes_series)

    print(edf_preprocessed_with_classes.head())
    proportion_class_1 = torch.sum(predicted_classes == 1).item() / len(predicted_classes)
    print(torch.sum(predicted_classes == 1))
    print(f"Proportion of predicted class 1: {proportion_class_1:.2f}") 
   # print(load_model('model_dir/model.pth'))
    #print(predict('aaaaaajy_s002_t000.edf', '01_tcp_ar'))
