import torch
from scipy.signal import butter, lfilter, filtfilt, iirnotch, resample, sosfilt, sosfiltfilt

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# Function to normalize a subset of leads
def normalize_leads(full_data, lead_indices):
    mean = torch.mean(full_data[:, lead_indices, :], dim=(0), keepdim=True)  # Mean across participants
    std = torch.std(full_data[:, lead_indices, :], dim=(0), keepdim=True)  # Std across participants
    full_data[:, lead_indices, :] = (full_data[:, lead_indices, :] - mean) / std  # Normalize selected leads
    return full_data

def normalize_per_patient(full_data):
    mean = full_data.mean(dim=(1, 2), keepdim=True)
    std = full_data.std(dim=(1, 2), keepdim=True)

    # add small epsilon in case of std being 0, i.e. in flat not-recorded ECGs
    return (full_data - mean) / (std + 1e-5) 


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs # half the sampling rate
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def butter_highpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    high_cutoff = cutoff / nyq
    sos = butter(order, high_cutoff, btype='highpass', output='sos')
    return sos

def highpass_filter(data, cut, fs, order=4):  # Default ECG sampling rate ~500 Hz
    sos = butter_highpass(cut, fs, order)
    return sosfiltfilt(sos, data)


def notch_filter(signal, fs, freq=50, quality=40):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal)


def resample_signal(signal, fs_original, fs_target=400):
    num_samples = int(len(signal) * fs_target / fs_original)
    return resample(signal, num_samples)
