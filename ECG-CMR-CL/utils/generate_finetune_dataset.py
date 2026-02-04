import argparse
import torch
import os
import pandas as pd
import numpy as np
import re

def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='/data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--output_dir', default='/data',
                        type=str, help='The output directory to save the ECG dataset for fine-tunning')

    return parser


def align_ECG_data_and_pheno_data(pheno_data: pd.DataFrame, split: str, args) -> np.array:

    os.chdir(args.ecg_data_dir)

    print(f"Current working directory: {os.getcwd()}")

    # filter the IDs if not in ECG path
    ECG_ids = torch.load(f"/data/ECG_ids_{split}.pt").numpy().astype(int)

    # load the ECG tensors
    ECG_leads = torch.load(f"ECG_leads_{split}_per_pat.pt").numpy()

    # Find which IDs from ECG_ids_train are missing in pheno_data
    missing_ids = [id_ for id_ in ECG_ids if id_ not in pheno_data["f.eid"].values]
    print(f"Missing IDs in {split} set: {missing_ids}")

    # Get the positions of missing IDs in ECG_ids_train
    missing_positions = [np.where(ECG_ids == missing_id)[0][0] for missing_id in missing_ids]
    print(f"Positions of missing IDs in ECG_ids_{split}: {missing_positions}")

    # Extract the missing positions in the ECG data and the IDs tensors
    # Remove the elements at the missing positions
    mask = np.ones(len(ECG_ids), dtype=bool)
    mask[missing_positions] = False
    ECG_leads = ECG_leads[mask]
    ECG_leads = torch.from_numpy(ECG_leads)
    ECG_ids = ECG_ids[mask]

    os.chdir("/gpfs")
    torch.save(ECG_leads, os.path.join(args.output_dir, f"ECG_leads_{split}"))
    torch.save(ECG_ids, os.path.join(args.output_dir, f"ECG_ids_{split}"))
    del ECG_leads, missing_positions, missing_ids

    return ECG_ids

def main(args):

    # Get the directory of the current Python file
    initial_file_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory
    parent_dir = "/gpfs"

    # Change the working directory to the parent directory
    os.chdir(parent_dir)

    print(f"Initial working directory: {initial_file_dir}")
    print(f"Current working directory: {os.getcwd()}")

    # Get all the diagnosis columns
    pheno_data = pd.read_csv('/phenotypes',
                        sep='\t', compression='gzip',
                        nrows=5
                        ) # age imaginig visit, sex, QRS duration, QRS num
    diagn_cols = [col for col in pheno_data.columns if re.search('41270', col)]
    diagn_cols.append('f.eid')
    diagn_cols.append('f.24100.2.0') # LV end diastolic volume
    diagn_cols.append('f.24101.2.0') # LV end systolic volume
    diagn_cols.append('f.24103.2.0') # LVEF

    # Load the TSV file (gzip compressed) with the diagnosis columnns
    pheno_data = pd.read_csv('/phenotypes',
                        sep='\t', compression='gzip',
                        usecols=diagn_cols,
                        )

    ECG_ids_train = align_ECG_data_and_pheno_data(pheno_data, split="train", args=args)
    ECG_ids_val = align_ECG_data_and_pheno_data(pheno_data, split="val", args=args)
    ECG_ids_test = align_ECG_data_and_pheno_data(pheno_data, split="test", args=args)

    # relevant cardiac endpoints
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

    for endpoint, codes in endpoint_codes.items():
        if endpoint in exclude_keys:
            pheno_data[endpoint] = pheno_data[codes]
            continue

        pheno_data[endpoint] = pheno_data.apply(
            lambda row: any(code in row.values for code in codes), axis=1
        )

    endpoint_labels = pheno_data[endpoint_codes.keys()]
    endpoint_labels = endpoint_labels.replace({False: 0, True: 1})

    endpoint_labels["f.eid"] = pheno_data['f.eid'].astype(int)

    train_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_train)]
    print(train_endpoint_labels.shape)

    val_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_val)]
    print(val_endpoint_labels.shape)

    test_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_test)]
    print(test_endpoint_labels.shape)

    print("endpoint_labels.shape[0]", endpoint_labels.shape[0])

    print("ECG_ids_test", ECG_ids_test[:10])
    train_endpoint_labels = train_endpoint_labels.set_index("f.eid").loc[ECG_ids_train]
    print(train_endpoint_labels.head())
    val_endpoint_labels = val_endpoint_labels.set_index("f.eid").loc[ECG_ids_val]
    test_endpoint_labels = test_endpoint_labels.set_index("f.eid").loc[ECG_ids_test]

    train_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_labels.csv"), index=False, header=False)
    val_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_val_labels.csv"), index=False, header=False)
    test_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_test_labels.csv"), index=False, header=False)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
