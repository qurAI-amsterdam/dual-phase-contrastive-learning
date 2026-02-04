import torch
from torch.utils.data import Dataset
import pandas as pd


class CLDataset(Dataset):
    def __init__(self, cmr_data_path, ecg_data_path, cmr_data_path_systole, labels_path,
                 train=True, cmr_transform=None, ecg_transform=None):
        self.cmr_data = torch.load(cmr_data_path, map_location=torch.device('cpu'))
        print("cmr_data_shape:", self.cmr_data.shape)
        self.cmr_transform = cmr_transform
        self.cmr_data_len = self.cmr_data.shape[0]

        self.ecg_data = torch.load(ecg_data_path, map_location=torch.device('cpu'))
        self.ecg_transform = ecg_transform
        print("ecg_data_shape:", self.ecg_data.shape)
        self.ecg_data_len = self.ecg_data.shape[0]

        if cmr_data_path_systole is not None:
            self.cmr_data_systole = torch.load(cmr_data_path_systole, map_location=torch.device('cpu'))
            assert self.cmr_data_len == self.cmr_data_systole.shape[0], "The length of the CMR dataset"
            " and the CMR systole dataset does not match."
            print("cmr_data_systole:", self.cmr_data_systole.shape)
        else:
            self.cmr_data_systole = False

        if labels_path is not None:
            if labels_path.endswith(".csv"):
                labels = pd.read_csv(labels_path, header=None).squeeze()  
                self.labels = torch.tensor(labels.values, dtype=torch.half).view(-1, 1)
            else:
                self.labels = torch.load(labels_path, map_location=torch.device('cpu'), weights_only=False, dtype=torch.half)
        else:
            self.labels = False

        assert self.cmr_data_len == self.ecg_data_len, "The length of the CMR dataset"
        " and the ECG dataset does not match."

    def __len__(self) -> int:
        return self.cmr_data_len

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        cmr_data = self.cmr_data[idx]
        ecg_data = self.ecg_data[idx]

        if self.cmr_transform:
            cmr_data = self.cmr_transform(cmr_data)

        if self.ecg_transform:
            ecg_data = self.ecg_transform(ecg_data)

        if self.cmr_data_systole is not False:
            cmr_data_systole = self.cmr_data_systole[idx]
            if self.cmr_transform:
                cmr_data_systole = self.cmr_transform(cmr_data_systole)
                return cmr_data, ecg_data, cmr_data_systole

        if self.labels is not False:
            labels = self.labels[idx]
            return cmr_data, ecg_data, labels

        return cmr_data, ecg_data 

