import argparse
import torch
import os
import pandas as pd
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='/data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--output_dir', default='/ECG_data',
                        type=str, help='The output directory to save the ECG dataset for fine-tunning')

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
        "LV_end_diastolic": 'f.24100.2.0',                                  # LV end diastolic volume
        "LV_end_systolic": 'f.24101.2.0',                                   # LV end systolic volume
        "LV_ejection_fraction": 'f.24103.2.0',                              # LVEF
    }

    exclude_keys = {
        "LV_end_diastolic",
        "LV_end_systolic",
        "LV_ejection_fraction"
    }
    for split in ["train", "val", "test"]:
        np.random.seed(42)

        # Load ECG tensor and labels
        train_labels = pd.read_csv(os.path.join(args.output_dir, f"ECG_fine_tune_{split}_labels.csv"), header=None)
        clinical_inputs = pd.read_csv(os.path.join("/home/abujalancegome/deep_risk/data", f"ECG_fine_tune_{split}_clinical_input.csv"), header=None)

        print("train_labels.head()", train_labels.head())
        print("train_labels.type()", type(train_labels))
        print("train_labels.columns()", train_labels.head())

        train_ECG_leads = torch.load(os.path.join(args.output_dir,f"ECG_leads_{split}"))
        train_ECG_leads = train_ECG_leads.squeeze(2)
        train_ECG_leads = train_ECG_leads.unsqueeze(1)
        torch.save(train_ECG_leads, os.path.join(args.output_dir, f"ECG_unbalanced_{split}_leads.pt"))
        # if split == "val":
        #    train_ECG_leads = train_ECG_leads.squeeze(1)
        print(train_ECG_leads.shape)

        for pos, endpoint_code in enumerate(endpoint_codes.keys()):

            # select the indexes with at least one positive case
            print(f"----- Processing {endpoint_code} at {split} -----:")
            if endpoint_code in exclude_keys:
                train_labels_clean = train_labels[pos].dropna()
                train_indices = train_labels_clean.index

                balanced_train_labels = train_labels[pos].iloc[train_indices].reset_index(drop=True)

                print(f"balanced_train_labels for {endpoint_code}:", balanced_train_labels.shape)
                print(f"number of true cases for {endpoint_code}:", balanced_train_labels.sum())
                balanced_train_labels.to_csv(os.path.join(args.output_dir, f"ECG_balanced_{endpoint_code}_{split}_labels.csv"), header=False, index=False)

                del balanced_train_labels

                balanced_train_ECG_leads = train_ECG_leads[train_indices]
                torch.save(balanced_train_ECG_leads, os.path.join(args.output_dir, f"ECG_balanced_{endpoint_code}_{split}_leads.pt"))
                del balanced_train_ECG_leads
                continue
            positive_mask = train_labels[pos] > 0
            negative_mask = ~positive_mask

            positive_indices = np.where(positive_mask)[0]
            negative_indices = np.where(negative_mask)[0]

            print(positive_indices[:10])
            print(len(positive_mask), len(positive_indices))

            # balance the dataset, removing extra negatives indices
            np.random.shuffle(negative_indices)
            negative_indices = negative_indices[:len(positive_indices)]

            # shuffle the positive and negative cases
            train_indices = np.append(positive_indices, negative_indices)
            np.random.shuffle(train_indices)

            # get balanced data and labels
            balanced_train_labels = train_labels[pos].iloc[train_indices].reset_index(drop=True)
            unbalanced_train_labels = train_labels[pos]
            print(f"balanced_train_labels for {endpoint_code}:", balanced_train_labels.shape)
            print(f"number of true cases for {endpoint_code}:", balanced_train_labels.sum())
            balanced_train_labels.to_csv(os.path.join(args.output_dir, f"ECG_balanced_{endpoint_code}_{split}_labels.csv"), header=False, index=False)
            unbalanced_train_labels.to_csv(os.path.join(args.output_dir, f"ECG_unbalanced_{endpoint_code}_{split}_labels.csv"), header=False, index=False)
            del balanced_train_labels, unbalanced_train_labels

            balanced_train_ECG_leads = train_ECG_leads[train_indices]
            torch.save(balanced_train_ECG_leads, os.path.join(args.output_dir, f"ECG_balanced_{endpoint_code}_{split}_leads.pt"))
            del balanced_train_ECG_leads

            # get balanced clinical input data
            balanced_clinical_inputs = clinical_inputs.iloc[train_indices].reset_index(drop=True)
            balanced_clinical_inputs.to_csv(os.path.join(args.output_dir, f"ECG_balanced_{endpoint_code}_{split}_clinical_input.csv"), header=False, index=False)
            clinical_inputs.to_csv(os.path.join(args.output_dir, f"ECG_unbalanced_{endpoint_code}_{split}_clinical_input.csv"), header=False, index=False)
            del balanced_clinical_inputs


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
