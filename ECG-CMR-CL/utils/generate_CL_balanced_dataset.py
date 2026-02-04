import argparse
import torch
import os
import pandas as pd
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--output_dir', default='/CL_3D',
                        type=str, help='The output directory to save the ECG dataset for CL with a more balanced dataset')
    parser.add_argument('--labels_dir', default='/ECG_data/',
                        type=str, help='The output directory to save the CMR dataset for CL with a more balanced dataset')

    return parser


def main(args):
    # Get the directory of the current Python file
    initial_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    parent_dir = "/gpfs"

    # Change the working directory to the parent directory
    os.chdir(parent_dir)

    print(f"Initial working directory: {initial_file_dir}")
    print(f"Current working directory: {os.getcwd()}")

    endpoint_codes = {
        "coronary_artery_disease": [f'I{num}' for num in range(200, 260)],  # ICD I20-I25
        "atrial_fibrilation": [f'I{num}' for num in range(480, 490)],       # ICD I48
        "sudden_cardiac_death": [f'I{num}' for num in range(460, 480)],     # ICD I46-I47
        "heart_failure": [f'I{num}' for num in range(500, 510)],            # ICD I50
        "myocardial_infarction": [f'I{num}' for num in range(250, 260)],    # ICD I25
        "cardiomyopathy": [f'I{num}' for num in range(420, 430)],           # ICD I42
    }

    for split in ["train", "val", "test"]:
        np.random.seed(42)

        # Load ECG tensor and labels
        train_labels = pd.read_csv(os.path.join(args.labels_dir, f"ECG_fine_tune_{split}_labels.csv"), header=None)

        print("train_labels.head()", train_labels.head())
        print("train_labels.type()", type(train_labels))
        print("train_labels.columns()", train_labels.head())

        train_ECG_leads = torch.load(os.path.join("/projects/prjs1252/CL_data/ECG_data/",f"CL_ECG_leads_diastole_{split}_3d.pt"))

        # train_ECG_leads = train_ECG_leads.squeeze(2)
        # train_ECG_leads = train_ECG_leads.unsqueeze(1)

        train_labels = train_labels.iloc[:train_ECG_leads.shape[0]]
        CMR_tensors = torch.load(os.path.join("/projects/prjs1252/CL_data/ECG_data/",f"CL_cmr_tensor_diastole_{split}_3d.pt"))
        # torch.save(train_ECG_leads, os.path.join(args.output_dir, f"ECG_unbalanced_{split}_leads.pt"))
        # if split == "val":
        #    train_ECG_leads = train_ECG_leads.squeeze(1)
        print(train_ECG_leads.shape)

        for pos, endpoint_code in enumerate(endpoint_codes.keys()):
            print(f"----- Processing {endpoint_code} at {split} -----:")

            # select the indexes with at least one positive case
            positive_mask = train_labels[pos] > 0
            # positive_rows = positive_mask.any(axis=1)
            negative_mask = ~positive_mask
            # negative_rows = ~positive_rows
            print("positive_mask:\n", positive_mask.head(11))
            print("negative_mask:\n", negative_mask.head(11))

            positive_indices = np.where(positive_mask)[0]
            negative_indices = np.where(negative_mask)[0]

            print("negative_indices[:10]:", negative_indices[:10])
            print("len(negative_indices):", len(negative_indices))

            print("positive_indices[:10]:", positive_indices[:10])
            # positive_indices = np.unique(positive_indices)
            positive_indices.sort()
            # print(len(positive_mask), len(positive_indices))

            # balance the dataset, removing extra negatives indices
            np.random.shuffle(negative_indices)
            if split == "train":
                negative_indices = negative_indices[:len(positive_indices)]
            else:
                negative_indices = negative_indices[:len(positive_indices) * 3]

            print("len(negative_indices) after balancing:", len(negative_indices))
            print("len(positive_indices):", len(positive_indices))

            # shuffle the positive and negative cases
            train_indices = np.append(positive_indices, negative_indices)
            np.random.shuffle(train_indices)

            # get balanced data and labels
            balanced_train_labels = train_labels[pos].iloc[train_indices].reset_index(drop=True)

            print(f"balanced_train_labels for {endpoint_code}:", balanced_train_labels.shape)
            print(f"number of true cases for {endpoint_code}:", balanced_train_labels.sum())
            balanced_train_labels.to_csv(os.path.join(args.output_dir, f"ECG_balanced_{endpoint_code}_diastole_{split}_labels.csv"), header=False, index=False)
            del balanced_train_labels

            balanced_train_ECG_leads = train_ECG_leads[train_indices]
            print("balanced_train_ECG_leads.shape:", balanced_train_ECG_leads.shape)
            torch.save(balanced_train_ECG_leads, os.path.join(args.output_dir, f"CL_ECG_leads_balanced_{endpoint_code}_diastole_{split}_3d.pt"))
            del balanced_train_ECG_leads

            balanced_CMR_tensor = CMR_tensors[train_indices]
            print("balanced_CMR_tensor.shape:", balanced_CMR_tensor.shape)
            torch.save(balanced_CMR_tensor, os.path.join(args.output_dir, f"CL_cmr_tensor_balanced_{endpoint_code}_diastole_{split}_3d.pt"))
            del balanced_CMR_tensor


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
