import pyabf
import scipy
import pyabf
from scipy.signal import savgol_filter
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from scipy.signal import welch
from scipy import signal
from scipy.signal import butter, lfilter


def check_single_pulse(abf):
  normC= abs(abf.sweepC) - abs(abf.sweepC[0])
  first_pulse = np.argmax(normC)

  for i in range(first_pulse, normC.shape[0]):
      if normC[i] == normC[first_pulse]:
        pass
      else:
        end_first_pulse = i
        break

  for i in range(end_first_pulse, normC.shape[0]):
      if normC[i] == normC[end_first_pulse]:
        pass
      else:
        print("new signal detected at", i)
        return False
        break
  return True

def low_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a low pass Butterworth filter.

    Parameters:
    - data : array_like, The data to filter.
    - cutoff : float, The cutoff frequency of the filter.
    - fs : float, The sampling rate of the data.
    - order : int, The order of the filter.

    Returns:
    - filtered_data : array_like, The filtered data.
    """
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def correct_abf_signal(abf):
    sweepynorm = []
    for i in range(abf.sweepCount):
      # Assuming you are working with the first channel and first sweep
      abf.setSweep(sweepNumber=i)
      baseline = abf.sweepY[:np.argmax(abs(abf.sweepC))].mean() # take before the peak to correct offset
      sweepynorm.append((abf.sweepY - baseline) )
    sweepynormed = np.vstack(sweepynorm)
    return sweepynormed

def smooth_abf_signal(Ysignal,  window_size = 21, poly_order = 3 ):
  smoothed_signal = []
  for i in range(Ysignal.shape[0]):
    smoothed_signal.append(savgol_filter(Ysignal[i,:], window_size, poly_order))
  smoothed_signal = np.vstack(smoothed_signal)
  return smoothed_signal

def norm_abf_signal(Ysignal, pulse_offset = 400, end_time = 6000):
    sweepynorm = []
    start_pulse = np.argmax(abs(abf.sweepC)) + pulse_offset

    Ysignal = Ysignal[:, start_pulse:end_time]
    for i in range(Ysignal.shape[0]):
      norm_max = max(abs(Ysignal[i,:]))
      print(norm_max)
      sweepynorm.append(Ysignal[i,:] / (norm_max + 0.000001) )

    sweepynorm = np.vstack(sweepynorm)
    return sweepynorm


def extract_features(signals, start_pulse_signal):
    features_list = []

    for s in signals:
        _, psd = welch(s)
        psd_norm = psd / np.sum(psd)
        # Calculate the statistical features for each signal


        fft_results = fft(s)

        # Get the absolute value to find the magnitude of the frequencies
        fft_magnitudes = np.abs(fft_results)

        # Since FFT output is symmetrical, we take the first half of the frequencies
        # which represent the frequency spectrum from 0 to the Nyquist frequency
        half_n_samples = signals.shape[1] // 2
        fft_magnitudes = fft_magnitudes[:half_n_samples]

        # Feature 1: The maximum magnitude (the peak in the FFT)
        max_magnitude = np.max(fft_magnitudes)

        # Feature 2: The frequency at which the maximum magnitude occurs
        max_freq_index = np.argmax(fft_magnitudes)


        features_dict = {
            'min': np.min(s),
            'max': np.max(s),
            'integral': np.trapz(-s),
            'time_onset_peak': np.argmin(np.diff(s)) - start_pulse_signal,
            'time_onset_max_peak': np.argmax(np.diff(s)) - start_pulse_signal,
            'time_max': np.argmax(s) - start_pulse_signal,
            'mean': np.mean(s),
            'median': np.median(s),
            'std_dev': np.std(s),
            'skewness': skew(s),
            'energy': np.sum(np.square(s)),
            'spectral_entropy': -np.sum(psd_norm * np.log2(psd_norm)),
            'peak_to_peak': np.ptp(s),
            'autocorr_feature': np.correlate(s, s, mode='full')[len(s)-1],
            'peaks_count_p1': len(signal.find_peaks(-s, prominence=100, distance=100)[0]),
            'peaks_count_p2': len(signal.find_peaks(-s, prominence=50, distance=200)[0]),
            'peaks_count_p3': len(signal.find_peaks(-s, prominence=100)[0]),
            'peaks_count_p4': len(signal.find_peaks(-s, prominence=50)[0]),
            'max_magnitude':max_magnitude,
            'max_freq_index':max_freq_index

        }
        features_list.append(features_dict)

    # Convert the list of feature dictionaries to a DataFrame
    features_df = pd.DataFrame(features_list)

    return features_df


