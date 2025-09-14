import os
import glob
import pandas as pd

def combine_datasets(base_path):
    train_files = glob.glob(os.path.join(base_path, "C*", "train*.csv"))
    train_dfs = []
    for file in train_files:
        df = pd.read_csv(file)
        country_code = os.path.basename(os.path.dirname(file))
        df["country_code_file"] = country_code
        train_dfs.append(df)
    train = pd.concat(train_dfs, ignore_index=True)

    test_files = glob.glob(os.path.join(base_path, "C*", "test*.csv"))
    test_dfs = []
    for file in test_files:
        df = pd.read_csv(file)
        country_code = os.path.basename(os.path.dirname(file))
        df["country_code_file"] = country_code
        test_dfs.append(df)
    test = pd.concat(test_dfs, ignore_index=True)

    return train, test

if __name__ == "__main__":
    base_path = "../Data/"
    train, test = combine_datasets(base_path)

    output_dir = "Data"
    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
