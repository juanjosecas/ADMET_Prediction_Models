import argparse
import os
import pickle

import numpy as np
import pandas as pd

from utils import data_preprocessing, load_features

def load_model(name: str):
    """Load a pickled model from the default locations."""
    for path in (f"models/{name}.pkl", f"{name}.pkl"):
        if os.path.exists(path):
            return pickle.load(open(path, "rb"))
    raise FileNotFoundError(f"Model {name}.pkl not found")


def predict(file_name: str, model_name: str):
    features = load_features()
    print(f"Prediction model for {model_name}")
    print(f"Loading the data set : {file_name}")

    df = pd.read_csv(file_name)
    df["bioclass"] = 1
    scaled_data = data_preprocessing(df)

    model = load_model(model_name)

    result_df = pd.DataFrame({"SMILES": scaled_data["SMILES"]})
    result_df[model_name] = model.predict(scaled_data[features].values)

    predict_proba = model.predict_proba(scaled_data[features].values)
    result_df["pred_prob"] = np.max(predict_proba, axis=1)

    print(f"Output for prediction of {model_name}\n{result_df}")
    result_df.to_csv(f"{model_name}_predict_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="example_train.csv", help="Path to the input file")
    parser.add_argument("--model_name", required=True, help="Model name without extension")
    args = parser.parse_args()

    predict(args.file_name, args.model_name)
