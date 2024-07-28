import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_edf(file_path):
    f = pyedflib.EdfReader(file_path)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signals = np.zeros((n, f.getNSamples()[0]))
    sampling_rate = f.getSampleFrequency(
        0
    )  # Assuming all signals have the same sampling rate

    for i in range(n):
        signals[i, :] = f.readSignal(i)

    f._close()
    del f
    return signals, signal_labels, sampling_rate


# this plots all signals together
def plot_edf(signals, signal_labels, sampling_rate, annotations):
    plt.figure(figsize=(12, 8))

    # Generate time values in seconds
    num_samples = signals.shape[1]
    time_values = np.arange(num_samples) / sampling_rate

    # Plot each signal and keep track of min and max values for setting y-axis limits
    min_val, max_val = float("inf"), float("-inf")
    for i, signal in enumerate(signals):
        plt.plot(time_values, signal, label=signal_labels[i], linewidth=0.7, alpha=0.7)
        min_val = min(min_val, np.min(signal))
        max_val = max(max_val, np.max(signal))

    # Highlight seizure events
    for _, row in annotations.iterrows():
        plt.axvspan(row["start_time"], row["stop_time"], color="red", alpha=0.2)

    plt.title("EDF Signals with Seizure Annotations")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.ylim(min_val - 0.1 * abs(min_val), max_val + 0.1 * abs(max_val))

    # Position legend to the right of the graph
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("eda_data/edf.png")
    plt.show()


def plot_edf_with_annotations(signals, signal_labels, sampling_rate, annotations):
    num_signals = signals.shape[0]
    time_values = np.arange(signals.shape[1]) / sampling_rate

    plt.figure(figsize=(12, 8))
    for i, signal in enumerate(signals):
        plt.subplot(num_signals, 1, i + 1)
        plt.plot(time_values, signal, label=signal_labels[i], linewidth=0.5, alpha=0.7)

        # Highlight seizure events
        for _, row in annotations.iterrows():
            plt.axvspan(row["start_time"], row["stop_time"], color="red", alpha=0.3)

        plt.title(signal_labels[i])
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig("eda_data/edf.png")
    plt.show()


# Replace 'your_edf_file.edf' with the path to your EDF file
edf_file_path = "eda_data/aaaaaajy_s001_t000.edf"
signals, signal_labels, sampling_rate = read_edf(edf_file_path)
annotations_path = "eda_data/annotations.csv"
annotations = pd.read_csv(annotations_path)

plot_edf(signals, signal_labels, sampling_rate, annotations)
# plot_edf_with_annotations(signals, signal_labels, sampling_rate, annotations)
