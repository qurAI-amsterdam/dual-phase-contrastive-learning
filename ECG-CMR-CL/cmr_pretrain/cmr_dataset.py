import torch
from torch.utils.data import Dataset
import pandas as pd


class CMRDataset(Dataset):
    def __init__(self, data_path, labels_path=None,
                 train=True, finetune=False, transform=None):
        self.data = torch.load(data_path, map_location=torch.device('cpu'))
        self.transform = transform

        if labels_path: # decide if labels are a torch file or a csv
            if labels_path.endswith(".csv"):
                labels = pd.read_csv(labels_path, header=None)
                self.labels = torch.tensor(labels.values, dtype=torch.half)
                if finetune:
                    labels = pd.read_csv(labels_path, header=None).squeeze()  
                    self.labels = torch.tensor(labels.values, dtype=torch.half).view(-1, 1)
            else:
                self.labels = torch.load(labels_path, map_location=torch.device('cpu'), weights_only=False, dtype=torch.half)
            # self.labels = pd.read_csv(labels_path) 
        else:
            raise Exception("Labels not provided. Not implemented self-supervised training.")
            

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        cmr_data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            cmr_data = self.transform(cmr_data)

        return cmr_data, label 


class ClincialDataset(Dataset):
    def __init__(self, data_path, labels_path=None, clinical_path=None,
                 train=True, finetune=False, transform=None, args=None):
        self.data = torch.load(data_path, map_location=torch.device('cpu'))
        self.transform = transform
        self.is_clinical_data = args.clinical_data

        if labels_path: # decide if labels are a torch file or a csv
            if labels_path.endswith(".csv"):
                labels = pd.read_csv(labels_path, header=None)
                self.labels = torch.tensor(labels.values, dtype=torch.half)
                if finetune:
                    labels = pd.read_csv(labels_path, header=None).squeeze()  
                    self.labels = torch.tensor(labels.values, dtype=torch.half).view(-1, 1)
            else:
                self.labels = torch.load(labels_path, map_location=torch.device('cpu'), weights_only=False, dtype=torch.half)
            # self.labels = pd.read_csv(labels_path) 
        else:
            raise Exception("Labels not provided. Not implemented self-supervised training.")

        if self.is_clinical_data:
            if clinical_path.endswith(".csv"):
                clinical_data = pd.read_csv(clinical_path, header=None)
                self.clinical_data = torch.tensor(clinical_data.values, dtype=torch.half)
            else:
                self.labels = torch.load(clinical_path, map_location=torch.device('cpu'),
                                         weights_only=False, dtype=torch.half)
        else:
            pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            ecg_data = self.transform(ecg_data)

        if self.is_clinical_data:
            clinical_data = self.clinical_data[idx]
            return ecg_data, clinical_data, label
        else:
            return ecg_data, label

        
