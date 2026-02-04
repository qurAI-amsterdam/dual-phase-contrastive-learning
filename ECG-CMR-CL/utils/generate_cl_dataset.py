import argparse
import torch
import os
import pandas as pd
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='/data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--cmr_data_dir', type=str, help='Path to the CMR data directory')
    parser.add_argument('--output_dir', type=str, help='The output directory to save the CL dataset')

    return parser

def main(args):
    ecg_ids = {}

    # the CL train, val, test splits will contain the same splits as in the ECG splits
    cmr_idx_ids = pd.read_csv('/home/abujalancegome/deep_risk/ukbb_cardiac/data/ids_cropped_images.csv',
                              header=None, names=['ids'])
    cmr_idx_ids['CMR idx'] = np.arange(0, len(cmr_idx_ids['ids']), 1)

    cmr_tensor = torch.load(os.path.join(args.cmr_data_dir, f"cmr_tensors_ts_all_diastole.pt"))

    for split in ["train", "val", "test"]:
        ecg_ids[split] = torch.load(os.path.join(args.ecg_data_dir, f"ECG_ids_{split}_per_pat.pt"))
        ecg_tensor = torch.load(os.path.join(args.ecg_data_dir, f"ECG_leads_{split}_per_pat.pt"))

        ecg_ids[split] = ecg_ids[split].tolist()
        ecg_ids_df = pd.DataFrame(ecg_ids[split], columns=['ids'])
        ecg_ids_df['ECG idx'] = np.arange(0, len(ecg_ids_df['ids']), 1)

        # these are the ids and the index to obtain the CMR data for CL
        CL_df = pd.merge(ecg_ids_df, cmr_idx_ids, on=['ids'])

        # get the ids in the same order as previously to ensure that the ECG and CMR patient index match
        CL_idx = torch.tensor(CL_df['CMR idx'], dtype=torch.long)
        cmr_split_tensor = cmr_tensor[CL_idx]

        # remove the idx from the ECG train split that do not have a CMR pair
        CL_idx = torch.tensor(CL_df['ECG idx'], dtype=torch.long)
        ecg_split_tensor = ecg_tensor[CL_idx]

        # Creates the output directory if doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)

        # Squezee dimension to get the channel dimension
        cmr_split_tensor = cmr_split_tensor.unsqueeze(1)
        ecg_split_tensor = ecg_split_tensor.squeeze(2)
        ecg_split_tensor = ecg_split_tensor.unsqueeze(1)
    
        print(f"CMR {split} tensor shape:", cmr_split_tensor.shape)
        print(f"ECG {split} tensor shape:", ecg_split_tensor.shape)

        torch.save(ecg_split_tensor, os.path.join(args.output_dir, f"CL_ECG_leads_diastole_{split}_3d_patients_norm.pt"))
        torch.save(cmr_split_tensor, os.path.join(args.output_dir, f"CL_cmr_tensor_diastole_{split}_3d_patients_norm.pt"))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
