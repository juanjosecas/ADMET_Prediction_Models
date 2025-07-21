import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

from utils import data_preprocessing

def split_dataset(df: pd.DataFrame, mode: str, frac: float):
    """Split data into train/test sets using the requested mode."""
    if mode == "stratify":
        train_data, test_data = train_test_split(
            df,
            random_state=42,
            test_size=frac,
            stratify=df["bioclass"],
            shuffle=True,
        )
    else:
        from dgllife.utils import ScaffoldSplitter

        df = df.rename(columns={"SMILES": "smiles"})
        train_set, _, test_set = ScaffoldSplitter.train_val_test_split(
            df, frac_train=1 - frac, frac_val=0, frac_test=frac
        )
        train_data = df.iloc[train_set.indices]
        test_data = df.iloc[test_set.indices]

    print(f"Size of train dataset: {len(train_data)}")
    print(f"Size of test  dataset: {len(test_data)}")
    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="smiles.csv", help="Path to the input file")
    parser.add_argument("--split_mode", choices=["scaffold", "stratify"], default="stratify", help="Dataset splitting method")
    parser.add_argument("--test_frac", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    args = parser.parse_args()

    df = pd.read_csv(args.file_name)
    print(f"Start data pre-processing for {args.file_name}")
    df = data_preprocessing(df)

    train_data, test_data = split_dataset(df, args.split_mode, args.test_frac)
    train_data.to_csv(f"train_{args.file_name}", index=False)
    test_data.to_csv(f"test_{args.file_name}", index=False)
