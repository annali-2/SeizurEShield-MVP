import pandas as pd
import os
from io import StringIO
import pyedflib
import numpy as np


class EEGMontage:
    def __init__(self):
        # 01_tcp_ar montage
        self.montage_pairs_01_tcp_ar = [
            ('EEG FP1-REF', 'EEG F7-REF'),
            ('EEG F7-REF', 'EEG T3-REF'),
            ('EEG T3-REF', 'EEG T5-REF'),
            ('EEG T5-REF', 'EEG O1-REF'),
            ('EEG FP2-REF', 'EEG F8-REF'),
            ('EEG F8-REF', 'EEG T4-REF'),
            ('EEG T4-REF', 'EEG T6-REF'),
            ('EEG T6-REF', 'EEG O2-REF'),
            ('EEG A1-REF', 'EEG T3-REF'),
            ('EEG T3-REF', 'EEG C3-REF'),
            ('EEG C3-REF', 'EEG CZ-REF'),
            ('EEG CZ-REF', 'EEG C4-REF'),
            ('EEG C4-REF', 'EEG T4-REF'),
            ('EEG T4-REF', 'EEG A2-REF'),
            ('EEG FP1-REF', 'EEG F3-REF'),
            ('EEG F3-REF', 'EEG C3-REF'),
            ('EEG C3-REF', 'EEG P3-REF'),
            ('EEG P3-REF', 'EEG O1-REF'),
            ('EEG FP2-REF', 'EEG F4-REF'),
            ('EEG F4-REF', 'EEG C4-REF'),
            ('EEG C4-REF', 'EEG P4-REF'),
            ('EEG P4-REF', 'EEG O2-REF')
        ]

        # 02_tcp_le montage
        self.montage_pairs_02_tcp_le = [
            ('EEG FP1-LE', 'EEG F7-LE'),
            ('EEG F7-LE', 'EEG T3-LE'),
            ('EEG T3-LE', 'EEG T5-LE'),
            ('EEG T5-LE', 'EEG O1-LE'),
            ('EEG FP2-LE', 'EEG F8-LE'),
            ('EEG F8-LE', 'EEG T4-LE'),
            ('EEG T4-LE', 'EEG T6-LE'),
            ('EEG T6-LE', 'EEG O2-LE'),
            ('EEG A1-LE', 'EEG T3-LE'),
            ('EEG T3-LE', 'EEG C3-LE'),
            ('EEG C3-LE', 'EEG CZ-LE'),
            ('EEG CZ-LE', 'EEG C4-LE'),
            ('EEG C4-LE', 'EEG T4-LE'),
            ('EEG T4-LE', 'EEG A2-LE'),
            ('EEG FP1-LE', 'EEG F3-LE'),
            ('EEG F3-LE', 'EEG C3-LE'),
            ('EEG C3-LE', 'EEG P3-LE'),
            ('EEG P3-LE', 'EEG O1-LE'),
            ('EEG FP2-LE', 'EEG F4-LE'),
            ('EEG F4-LE', 'EEG C4-LE'),
            ('EEG C4-LE', 'EEG P4-LE'),
            ('EEG P4-LE', 'EEG O2-LE')
        ]

        # 03_tcp_ar_a montage
        # missing ('EEG A1-REF', 'EEG T3-REF'), ('EEG T4-REF', 'EEG A2-REF'), 
        # compared to 01_tcp_ar
        self.montage_pairs_03_tcp_ar_a = [
            ('EEG FP1-REF', 'EEG F7-REF'),
            ('EEG F7-REF', 'EEG T3-REF'),
            ('EEG T3-REF', 'EEG T5-REF'),
            ('EEG T5-REF', 'EEG O1-REF'),
            ('EEG FP2-REF', 'EEG F8-REF'),
            ('EEG F8-REF', 'EEG T4-REF'),
            ('EEG T4-REF', 'EEG T6-REF'),
            ('EEG T6-REF', 'EEG O2-REF'),
            ('EEG T3-REF', 'EEG C3-REF'),
            ('EEG C3-REF', 'EEG CZ-REF'),
            ('EEG CZ-REF', 'EEG C4-REF'),
            ('EEG C4-REF', 'EEG T4-REF'),
            ('EEG FP1-REF', 'EEG F3-REF'),
            ('EEG F3-REF', 'EEG C3-REF'),
            ('EEG C3-REF', 'EEG P3-REF'),
            ('EEG P3-REF', 'EEG O1-REF'),
            ('EEG FP2-REF', 'EEG F4-REF'),
            ('EEG F4-REF', 'EEG C4-REF'),
            ('EEG C4-REF', 'EEG P4-REF'),
            ('EEG P4-REF', 'EEG O2-REF')
        ]

        # 04_tcp_le_a
        # missing ('EEG A1-LE', 'EEG T3-LE'), ('EEG T4-LE', 'EEG A2-LE')
        # compared to 02_tcp_le 
        self.montage_pairs_04_tcp_le_a = [
            ('EEG FP1-LE', 'EEG F7-LE'),
            ('EEG F7-LE', 'EEG T3-LE'),
            ('EEG T3-LE', 'EEG T5-LE'),
            ('EEG T5-LE', 'EEG O1-LE'),
            ('EEG FP2-LE', 'EEG F8-LE'),
            ('EEG F8-LE', 'EEG T4-LE'),
            ('EEG T4-LE', 'EEG T6-LE'),
            ('EEG T6-LE', 'EEG O2-LE'),
            ('EEG T3-LE', 'EEG C3-LE'),
            ('EEG C3-LE', 'EEG CZ-LE'),
            ('EEG CZ-LE', 'EEG C4-LE'),
            ('EEG C4-LE', 'EEG T4-LE'),
            ('EEG FP1-LE', 'EEG F3-LE'),
            ('EEG F3-LE', 'EEG C3-LE'),
            ('EEG C3-LE', 'EEG P3-LE'),
            ('EEG P3-LE', 'EEG O1-LE'),
            ('EEG FP2-LE', 'EEG F4-LE'),
            ('EEG F4-LE', 'EEG C4-LE'),
            ('EEG C4-LE', 'EEG P4-LE'),
            ('EEG P4-LE', 'EEG O2-LE')
        ]

        # Create a dictionary for montage
        self.montage_dict = {
            '01_tcp_ar': self.montage_pairs_01_tcp_ar,
            '02_tcp_le': self.montage_pairs_02_tcp_le,
            '03_tcp_ar_a': self.montage_pairs_03_tcp_ar_a,
            '04_tcp_le_a': self.montage_pairs_04_tcp_le_a,
        }

    def read_edf_to_dataframe(self, file_path):
        try:
            f = pyedflib.EdfReader(file_path)

            n = f.signals_in_file
            signal_labels = f.getSignalLabels()

            # Initialize a dictionary to hold the signal data and their respective time axes
            data = {}
            time_axes = []

            max_length = 0

            for i in range(n):
                signal_label = signal_labels[i]
                fs = f.getSampleFrequency(i)
                signal_data = f.readSignal(i)
                num_samples = len(signal_data)
                duration = num_samples / fs
                time_axis = np.array([j / fs for j in range(num_samples)])

                data[signal_label] = signal_data
                data[f'timestamp_{signal_label}'] = time_axis

                time_axes.extend(time_axis)

                if num_samples > max_length:
                    max_length = num_samples

            # Determine the common timestamps
            common_timestamps = np.unique(np.round(time_axes, decimals=6))

            # Create a DataFrame to hold the common timestamps
            df = pd.DataFrame({'timestamp': common_timestamps})

            # Align each signal's data to the common timestamps
            for i in range(n):
                signal_label = signal_labels[i]
                time_axis = data.pop(f'timestamp_{signal_label}')
                signal_data = data.pop(signal_label)

                # Create a DataFrame for the current signal
                signal_df = pd.DataFrame({
                    'timestamp': time_axis,
                    signal_label: signal_data
                })

                # Merge with the common timestamps DataFrame
                df = pd.merge_asof(df, signal_df, on='timestamp', direction='nearest')

            df['file_path'] = file_path  # Add file path column
            return df
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    def compute_differential_signals(self, data, montage_pairs):
        """
        Compute differential signals for each montage pair in the provided data.

        Args:
        - data (pd.DataFrame): DataFrame containing EEG data columns.
        - montage_pairs (list of tuples): List of montage pairs (channel pairs) to compute differential signals.

        Returns:
        - pd.DataFrame: DataFrame with computed differential signals and additional columns.
        """
        # Initialize an empty DataFrame to store differential signals
        differential_df = pd.DataFrame()

        # Compute differential signals for each montage pair
        for pair in montage_pairs:
            electrode_1, electrode_2 = pair
            if electrode_1 in data.columns and electrode_2 in data.columns:
                differential_signal = data[electrode_1] - data[electrode_2]
                column_name = f"{electrode_1.split()[1].split('-')[0]}-{electrode_2.split()[1].split('-')[0]}"  # Extract and format the desired column name
                differential_df[column_name] = differential_signal
            else:
                print(f"Warning: Columns {electrode_1} or {electrode_2} do not exist in the DataFrame and will be skipped.")

        # Append original columns from data to differential_df
        for column in ['timestamp', 'file_path', 'label', 'confidence']:
            if column in data.columns:
                differential_df[column] = data[column]

        return differential_df