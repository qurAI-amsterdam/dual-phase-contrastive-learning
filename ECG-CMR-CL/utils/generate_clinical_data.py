import argparse
import torch
import os
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder

def get_args_parser():
    parser = argparse.ArgumentParser('Generate the CL dataset based on the ECG and the CMR data paths',
                                     add_help=False)

    parser.add_argument('--ecg_data_dir', default='data', type=str, help='Path to the ECG data directory')
    parser.add_argument('--output_dir', default='/data',
                        type=str, help='The output directory to save the ECG dataset for fine-tunning')

    return parser


def align_ECG_data_and_pheno_data(pheno_data: pd.DataFrame, split: str, args) -> np.array:

    os.chdir(args.ecg_data_dir)

    # filter the IDs if not in ECG path
    ECG_ids = torch.load(f"ECG_ids_{split}.pt").numpy().astype(int)

    # load the ECG tensors
    ECG_leads = torch.load(f"ECG_leads_{split}.pt").numpy()

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
    pheno_data = pd.read_csv('phenotypes',
                        sep='\t', compression='gzip',
                        nrows=0
                        )

    # diagnostic columns corresponding to clinical aspects such as sex or smoking
    diagn_cols = ['f.31.0.0', 'f.21003.2.0', 'f.20116.2.0', 'f.4080.2.0', 'f.4080.0.0',
                  'f.23400.0.0', 'f.23406.0.0', 'f.23405.0.0', 'f.21001.2.0', 'f.21001.0.0',
                  'f.1558.2.0']
    diagn_cols.append('f.eid')

    print(diagn_cols)

    # Load the TSV file (gzip compressed) with the diagnosis columnns
    pheno_data = pd.read_csv('phenotypes',
                        sep='\t', compression='gzip',
                        usecols=diagn_cols,
                        )
    
    print("pheno_data", pheno_data)
    

    ECG_ids_train = align_ECG_data_and_pheno_data(pheno_data, split="train", args=args)
    ECG_ids_val = align_ECG_data_and_pheno_data(pheno_data, split="val", args=args)
    ECG_ids_test = align_ECG_data_and_pheno_data(pheno_data, split="test", args=args)

    # # relevant clinical codes 
    # endpoint_codes = {
    #      "clinical_codes": diagn_cols,
    # }

    # for endpoint, codes in endpoint_codes.items():
    #     pheno_data[endpoint] = pheno_data.apply(
    #         lambda row: any(code in row.values for code in codes), axis=1
    #     )

    endpoint_labels = pheno_data
    # endpoint_labels = endpoint_labels.replace({False: 0, True: 1})

    endpoint_labels["f.eid"] = pheno_data['f.eid'].astype(int)

    train_endpoint_labels = endpoint_labels[endpoint_labels.loc[:, "f.eid"].isin(ECG_ids_train)]
    print(train_endpoint_labels.shape)

    print("train_endpoint_labels.head()")
    print(train_endpoint_labels.head())

    # Count rows with at least one NaN
    nan_rows_count = train_endpoint_labels.isna().any(axis=1).sum()

    # Total number of rows
    total_rows = len(train_endpoint_labels)

    # Percentage of rows with NaNs
    nan_percentage = (nan_rows_count / total_rows) * 100

    # Count number of NaN by columns
    nan_per_column = train_endpoint_labels.isna().sum()

    # Percentage of NaNs per column
    nan_per_col_percentage = (nan_per_column / total_rows) * 100

    print(f"Rows with NaNs: {nan_rows_count}")
    print(f"Percentage of rows with NaNs: {nan_percentage:.2f}%")
    print(f"Percentage of rows with NaNs: {nan_per_col_percentage.round(2)}%")

    # Impute missing values
    train_endpoint_labels['f.21003.2.0'] = train_endpoint_labels['f.21003.2.0'].fillna(train_endpoint_labels['f.21003.2.0'].mean())
    train_endpoint_labels['f.20116.2.0'] = train_endpoint_labels['f.20116.2.0'].fillna(train_endpoint_labels['f.20116.2.0'].median())
    train_endpoint_labels['f.21001.2.0'] = train_endpoint_labels['f.21001.2.0'].fillna(train_endpoint_labels['f.21001.0.0'])
    train_endpoint_labels['f.21001.2.0'] = train_endpoint_labels['f.21001.2.0'].fillna(train_endpoint_labels['f.21001.2.0'].mean())
    train_endpoint_labels['f.4080.2.0'] = train_endpoint_labels['f.4080.2.0'].fillna(train_endpoint_labels['f.4080.0.0'])
    train_endpoint_labels['f.4080.2.0'] = train_endpoint_labels['f.4080.2.0'].fillna(train_endpoint_labels['f.4080.2.0'].mean())

    train_endpoint_labels['f.1558.2.0'] = train_endpoint_labels['f.1558.2.0'].fillna(6)

    train_endpoint_labels = train_endpoint_labels.drop(columns=['f.4080.0.0', 'f.21001.0.0', 'f.23400.0.0', 'f.23405.0.0', 'f.23406.0.0'])

    # Count rows with at least one NaN
    nan_rows_count = train_endpoint_labels.isna().any(axis=1).sum()

    # Total number of rows
    total_rows = len(train_endpoint_labels)

    # Percentage of rows with NaNs
    nan_percentage = (nan_rows_count / total_rows) * 100

    # Count number of NaN by columns
    nan_per_column = train_endpoint_labels.isna().sum()

    # Percentage of NaNs per column
    nan_per_col_percentage = (nan_per_column / total_rows) * 100

    print(f"Percentage of rows with NaNs: {nan_percentage:.2f}%")
    print(f"Percentage of rows with NaNs:\n {nan_per_col_percentage.round(2)}%")

    encoder = OneHotEncoder(sparse=False, dtype=int)  # sparse=True returns a sparse matrix
    train_endpoint_labels['f.1558.2.0'] = train_endpoint_labels['f.1558.2.0'].astype(int)
    encoded_array = encoder.fit_transform(train_endpoint_labels[['f.1558.2.0']])

    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['f.1558.2.0']), index=train_endpoint_labels.index)
    print("'f.1558.2.0'\n:", train_endpoint_labels['f.1558.2.0'].head())
    train_endpoint_labels = train_endpoint_labels.drop(columns=['f.1558.2.0'])
    train_endpoint_labels = pd.concat([train_endpoint_labels, encoded_df], axis=1)
    print("train_endpoint_labels\n:", train_endpoint_labels.head())
    print("train_endpoint_labels\n:", train_endpoint_labels.tail())
    train_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_clinical_input.csv"), index=False, header=False)


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

    # Impute missing values
    val_endpoint_labels['f.21003.2.0'] = val_endpoint_labels['f.21003.2.0'].fillna(train_endpoint_labels['f.21003.2.0'].mean())
    val_endpoint_labels['f.20116.2.0'] = val_endpoint_labels['f.20116.2.0'].fillna(val_endpoint_labels['f.20116.2.0'].median())
    val_endpoint_labels['f.21001.2.0'] = val_endpoint_labels['f.21001.2.0'].fillna(val_endpoint_labels['f.21001.0.0'])
    val_endpoint_labels['f.21001.2.0'] = val_endpoint_labels['f.21001.2.0'].fillna(train_endpoint_labels['f.21001.2.0'].mean())
    val_endpoint_labels['f.4080.2.0'] = val_endpoint_labels['f.4080.2.0'].fillna(val_endpoint_labels['f.4080.0.0'])
    val_endpoint_labels['f.4080.2.0'] = val_endpoint_labels['f.4080.2.0'].fillna(train_endpoint_labels['f.4080.2.0'].mean())

    val_endpoint_labels['f.1558.2.0'] = val_endpoint_labels['f.1558.2.0'].fillna(6)

    encoder = OneHotEncoder(sparse=False, dtype=int)  # sparse=True returns a sparse matrix
    encoded_array = encoder.fit_transform(val_endpoint_labels[['f.1558.2.0']])

    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['f.1558.2.0']), index=val_endpoint_labels.index)
    val_endpoint_labels = val_endpoint_labels.drop(columns=['f.1558.2.0'])
    val_endpoint_labels = pd.concat([val_endpoint_labels, encoded_df], axis=1)

    val_endpoint_labels = val_endpoint_labels.drop(columns=['f.4080.0.0', 'f.21001.0.0', 'f.23400.0.0', 'f.23405.0.0', 'f.23406.0.0'])


    test_endpoint_labels['f.21003.2.0'] = test_endpoint_labels['f.21003.2.0'].fillna(train_endpoint_labels['f.21003.2.0'].mean())
    test_endpoint_labels['f.20116.2.0'] = test_endpoint_labels['f.20116.2.0'].fillna(test_endpoint_labels['f.20116.2.0'].median())
    test_endpoint_labels['f.21001.2.0'] = test_endpoint_labels['f.21001.2.0'].fillna(test_endpoint_labels['f.21001.0.0'])
    test_endpoint_labels['f.21001.2.0'] = test_endpoint_labels['f.21001.2.0'].fillna(train_endpoint_labels['f.21001.2.0'].mean())
    test_endpoint_labels['f.4080.2.0'] = test_endpoint_labels['f.4080.2.0'].fillna(test_endpoint_labels['f.4080.0.0'])
    test_endpoint_labels['f.4080.2.0'] = test_endpoint_labels['f.4080.2.0'].fillna(train_endpoint_labels['f.4080.2.0'].mean())

    test_endpoint_labels['f.1558.2.0'] = test_endpoint_labels['f.1558.2.0'].fillna(6)

    encoder = OneHotEncoder(sparse=False, dtype=int)  # sparse=True returns a sparse matrix
    encoded_array = encoder.fit_transform(test_endpoint_labels[['f.1558.2.0']])

    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['f.1558.2.0']), index=test_endpoint_labels.index)
    test_endpoint_labels = test_endpoint_labels.drop(columns=['f.1558.2.0'])
    test_endpoint_labels = pd.concat([test_endpoint_labels, encoded_df], axis=1)

    test_endpoint_labels = test_endpoint_labels.drop(columns=['f.4080.0.0', 'f.21001.0.0', 'f.23400.0.0', 'f.23405.0.0', 'f.23406.0.0'])

    # train_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_clinical_input.csv"), index=False, header=False)
    val_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_val_clinical_input.csv"), index=False, header=False)
    test_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_test_clinical_input.csv"), index=False, header=False)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    # main(args)
    train_endpoint_labels = pd.read_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_clinical_input.csv"), header=None)
    print(train_endpoint_labels.head())
    train_endpoint_labels = train_endpoint_labels.drop(columns=[0])
    print(train_endpoint_labels.head())

    train_endpoint_labels.to_csv(os.path.join(args.output_dir, "ECG_fine_tune_train_clinical_input.csv"), index=False, header=False)
