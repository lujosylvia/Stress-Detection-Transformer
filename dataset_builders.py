import pandas as pd
import random
from sklearn.discriminant_analysis import StandardScaler
from scipy.signal import savgol_filter

from helpers import change_freq, load_data_file

#
# Uses z-scale normalization to normalize signals
#
def normalize_1D_signal(signal: list):
  scaler = StandardScaler()
  normalized_signal = scaler.fit_transform(signal.reshape(-1, 1))
  return normalized_signal.ravel()

#
# Uses the Savitzky-Golay filter to smooth a signal.
# Defaults to a window of 1000 datapoints and a polyorder of 3.
#
def smooth_1D_signal(signal: list, window: int = 1000, polyorder: int = 3):
  return savgol_filter(signal, window, polyorder)

#
# Combines the smoothing and normalization of a signal
#
def clean_1D_signal(signal: list, window: int = 3000, polyorder: int = 3):
  smoothed_signal = smooth_1D_signal(signal, window, polyorder)
  return normalize_1D_signal(smoothed_signal)

#
# Builds chest dataframes from each subject.
# Output is a dictionary where the key is the subject key and the value is the dataframe.
#
def build_chest_dataframe(pkl_path: str, start_index: int = 1500000):
  _df = pd.DataFrame()
  _data = load_data_file(pkl_path)
  _chest_data = _data['signal']['chest']

  _data_keys = _chest_data.keys()

  for key in _data_keys:
    sliced_data = _chest_data[key][start_index:]

    if _chest_data[key].shape[1] > 1:
      for i in range(_chest_data[key].shape[1]):
        _df[f"{key}_{i+1}"] = clean_1D_signal(sliced_data[:, i], window=30000)
    else:
      _df[key] = clean_1D_signal(sliced_data.ravel())

  _df['label'] = [0 if label != 2 and label != 6 else 1 for label in _data['label']][start_index:]

  return _df

#
# Builds wrist dataframes from each subject.
# Output is a dictionary where the key is the subject key and the value is the dataframe.
#
def build_wrist_dataframe(pkl_path: str, target_frequency=32, start_index: int = 75000):
  _df = pd.DataFrame()
  _data = load_data_file(pkl_path)
  _wrist_data = _data['signal']['wrist']

    # Handle ACC on its own - it is already sampled at 32Hz and has 3 columns.
  for i in range(_wrist_data['ACC'].shape[1]):
    signal_data = _wrist_data['ACC'][start_index:, i]
    _df[f"ACC_{i+1}"] = clean_1D_signal(signal_data)

  # Handle BVP - it is sampled at 64Hz and needs to be downsampled to 32Hz.
  _df['BVP'] = clean_1D_signal(change_freq(_wrist_data['BVP'].ravel(), original_frequency=64, target_frequency=target_frequency)[start_index:])

  # Handle EDA - it is sampled at 4Hz and needs to be upsampled to 32Hz.
  _df['EDA'] = clean_1D_signal(change_freq(_wrist_data['EDA'].ravel(), original_frequency=4, target_frequency=target_frequency)[start_index:])

  # Handle TEMP - it is sampled at 4Hz and needs to be upsampled to 32Hz.
  _df['TEMP'] = clean_1D_signal(change_freq(_wrist_data['TEMP'].ravel(), original_frequency=4, target_frequency=target_frequency)[start_index:])

  # Handle label - it is sampled at 700Hz and needs to be downsampled to 32Hz.
  label_data = change_freq(_data['label'], original_frequency=700, target_frequency=target_frequency)[start_index:]

  _df = _df.iloc[:len(label_data)]
  _df['label'] = [0 if label != 2 and label != 6 else 1 for label in label_data.astype(int)]

  return _df

#
# splits a dataset into training and testing sets.
#
def build_test_train_set(_df: pd.DataFrame, train_size: float = 0.8):
  split_idx = int(len(_df) * train_size)

  train_df, test_df = _df.iloc[:split_idx], _df.iloc[split_idx:]

  return (train_df, test_df)


#
# splits dataset into a K-fold dataset. Subjects are randomly assigned to each fold.
#
def split_dataset(subjects: list, splits: int = 5):
    subject_shuffle = subjects[:]
    random.shuffle(subject_shuffle)

    entries_per_set = len(subject_shuffle) // splits
    dataset = []
    for i in range(splits):
        start = i * entries_per_set
        new_set = subject_shuffle[start:start + entries_per_set]
        dataset.append(new_set)

    return dataset