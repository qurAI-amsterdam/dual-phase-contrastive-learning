import os
import xml.etree.ElementTree as ET
import torch
import numpy as np
from pre_process_ecg_utils import notch_filter, bandpass_filter, leads, highpass_filter
from scipy.signal import savgol_filter
import pandas as pd

initial_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(initial_file_dir)))

print(initial_file_dir)
print(parent_dir)

os.chdir(parent_dir)

# Make directory the path with the ECG data
ecg_folder_path = ""
withdrawn_part = ""
withdrawn_df = pd.read_csv(withdrawn_part, header=None)
list_withdrawn = withdrawn_df[0].to_list()

tensors = [] # List to store ECG tensors
ids = [] # list to store the IDs

tensor_ECG_original = []
original_waveform = []

def extract_waveform_data(lead):
    # Find the <WaveformData> element with the specific lead
    waveform_element = root.find(f"StripData/WaveformData[@lead='{lead}']")
    if waveform_element is not None:
        # Extract the text content (comma-separated values)
        waveform_text = waveform_element.text.strip()
        # Convert the string to a list of floats
        waveform_values = [float(x) for x in waveform_text.split(',')]
        assert len(waveform_values) == 5000
        return waveform_values
    return print(f"waveform not found in {file_path} for lead {lead}")



# fs for pre-processing
fs_original = 500 # ECG recordings sampling frequency of 500 Hz

count = 0
# List all files in the folder
for filename in os.listdir(ecg_folder_path):
    # exclude the participants that have withdrawn from study participation
    id_ = filename.split('_')[0]
    if id_ in list_withdrawn:
        print(f"Participant with id {id_} have withdrawn from study participation.")
        continue

    # Check if the file has a .xml extension
    waveforms_file = []
    original_waveform = []
    if filename.endswith('.xml'):
        file_path = os.path.join(ecg_folder_path, filename)
        
        # Parse the XML file
        try:
            tree = ET.parse(file_path)
            
            root = tree.getroot()

            # Get all the 'lead' attributes from the <WaveformData> elements
            for lead in leads:
                waveform_values = np.array(extract_waveform_data(lead))

                # Apply pre-processing to the ECG lead https://www.nature.com/articles/s44161-024-00564-3

                # 1. Apply high pass filter
                # low_pass_filtered_lead = bandpass_filter(waveform_values, lowcut=0.5,
                #                                         highcut=100, fs=fs_original)
                high_pass_filtered_lead = highpass_filter(waveform_values, cut=0.5,
                                                          fs=fs_original)

                # 2. Apply notch filter (60 Hz)
                notch_filter_lead = notch_filter(high_pass_filtered_lead, fs_original, freq=60)

                # 3. Apply Savitzky-Golay filter
                pre_processed_lead = savgol_filter(notch_filter_lead, window_length=15, polyorder=3)

                waveforms_file.append(pre_processed_lead)
                original_waveform.append(waveform_values)

            file_tensor = torch.tensor(np.array(waveforms_file))
            original_ECG = torch.tensor(np.array(original_waveform))
            assert file_tensor.shape == torch.Size([12, 5000])
            tensors.append(file_tensor)
            tensor_ECG_original.append(original_ECG)
            ids.append(id_)

        except ET.ParseError as e:
            print(f"Error parsing {filename}: {e}")


combined_tensor = torch.stack(tensors)
original_ECG = torch.stack(tensor_ECG_original)

os.chdir(initial_file_dir)

print(type(combined_tensor))
print(combined_tensor.shape)
print(combined_tensor.dtype)

# full_data_ECGs_w_IDs = {"ECG_tensors": combined_tensor, "IDs": ids}
# full_data_ECGs_w_IDs = {"ECG_tensors": combined_tensor, "ECG_originals": original_ECG, "IDs": ids}
full_data_ECGs = {"ECG_tensors": combined_tensor, "IDs": ids}

torch.save(full_data_ECGs, "full_data_ECGs_w_IDs.pth")
