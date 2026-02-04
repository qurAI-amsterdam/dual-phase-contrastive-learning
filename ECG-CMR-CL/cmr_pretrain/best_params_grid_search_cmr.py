import os
import pandas as pd

base_folder = "test_dir"

best_loss = float('inf')
# for filename in os.listdir(ecg_folder_path):

for subfolder in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, subfolder)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.endswith(".pth"):
                loss_value = float(file.split('-')[3])
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_checkpoint = file


print("best checkpopint:", file)
