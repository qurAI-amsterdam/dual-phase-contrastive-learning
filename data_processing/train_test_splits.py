import torch
from sklearn.model_selection import train_test_split
from pre_process_ecg_utils import normalize_leads, normalize_per_patient
import numpy as np

# Load the full dataset: 10-s, 12-lead ECG, original samplig rate frequency 500 Hz
loaded_data = torch.load("/")

participant_ids = loaded_data["IDs"]  # List of participant IDs
full_data = loaded_data["ECG_tensors"].to(dtype=torch.float32)  # Shape: (N, 12, 5000)
print(f"Full dataset shape & dtype: {full_data.shape} & {full_data.dtype}")
# Calculate mean and standard deviation across participants

# Define lead indices
einthoven_idx = [0, 1, 2]  # I, II, III
goldberg_idx = [3, 4, 5]   # aVR, aVL, aVF
wilson_idx = [6, 7, 8, 9, 10, 11]  # V1-V6

# # Apply normalization separately for each lead group
# full_data = normalize_leads(full_data, einthoven_idx)
# full_data = normalize_leads(full_data, goldberg_idx)
# full_data = normalize_leads(full_data, wilson_idx)

# Apply Per-Patient Normalization
full_data = normalize_per_patient(full_data)

# Convert PyTorch tensor to NumPy for sklearn
full_data_np = full_data.numpy()
participant_ids_np = np.array(participant_ids)

# Split into train (70%) and temp (30%) which will be further split into val and test
train_np, temp_data, train_ids, temp_ids = train_test_split(
    full_data_np, participant_ids_np, test_size=0.3, random_state=42)

# Split temp into validation and test
val_np, test_np, val_ids, test_ids = train_test_split(
    temp_data, temp_ids, test_size=0.5, random_state=42)

# Convert back to PyTorch tensors and unsqueeze to get the correct tensor shapes
train_data = torch.tensor(train_np).unsqueeze(2)
val_data = torch.tensor(val_np).unsqueeze(2)
test_data = torch.tensor(test_np).unsqueeze(2)

train_ids = torch.tensor(train_ids.astype(int))
val_ids = torch.tensor(val_ids.astype(int))
test_ids = torch.tensor(test_ids.astype(int))

# Print shapes
print(f"Train set shape: {train_data.shape}")
print(f"Validation set shape: {val_data.shape}")
print(f"Test set shape: {test_data.shape}")

# Save the datasets
# torch.save(train_data, "ECG_leads_train.pt")
# torch.save(val_data, "ECG_leads_val.pt")
# torch.save(test_data, "ECG_leads_test.pt")

torch.save(train_data, "/")
torch.save(val_data, "/")
torch.save(test_data, "/")

# Save the IDs
torch.save(train_ids, "ECG_ids_train.pt")
torch.save(val_ids, "ECG_ids_val.pt")
torch.save(test_ids, "ECG_ids_test.pt")
