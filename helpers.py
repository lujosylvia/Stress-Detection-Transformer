import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

ORIGINAL_SAMPLING_RATE = {
  'BVP': 64,
  'EDA': 4,
  'TEMP': 4,
}

#
# Loads a .pkl file associated with the WESAD dataset.
# Takes in a relative path to the file to load.
#
def load_data_file(file_path: str):
  with open(file_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

  return data

#
# Updates the sampling frequency of a signal using interpolation.
# The original frequency defaults to 4 Hz, and the target frequency defaults to 32 Hz.
#
def change_freq(_data: np.array, original_frequency: int = 4, target_frequency: int = 32):
    sampling_factor = target_frequency / original_frequency

    new_length = int(len(_data) * sampling_factor)
    new_index = np.linspace(0, len(_data) - 1, new_length)

    return np.interp(new_index, np.arange(len(_data)), _data)


#
# Uses seaborn to plot a signal associated with a subject.
#
def display_signal(dataset: dict, subject: str = 'S3', signal: str = 'ACC'):
  data = pd.DataFrame({'Signal': dataset[subject][signal]})
  data['Time'] = data.index

  plt.figure(figsize=(10,4))

  sns.lineplot(x="Time", y="Signal", data=data)
  plt.title(f"Signal {signal} - Subject {subject}")

  plt.show()

#
# Displays all of the signals associated with a subject.
#
def display_subject_trial_signals(dataset: dict, subject: str = 'S3'):
  for key in dataset[subject].columns:
    display_signal(dataset, subject, key)