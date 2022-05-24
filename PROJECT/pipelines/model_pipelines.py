import os
import sys

sys.path.append("../")
os.chdir("../")

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import pandas as pd
import pickle as pkl

from data_pipelines import get_data_pipelines
from config import raw_data_full_path, model_path, mlp_hyperparams


def create_model(
    datapath: str,
    data_pipeline: Pipeline,
    output: str,
    model: MLPRegressor,
    target: str,
):
    df = pd.read_csv(datapath)
    X = df.drop(columns=[target])
    y = df[target]
    model_pipeline = Pipeline(
        [("data_preproc", data_pipeline), ("model_pipeline", model)]
    )
    model_pipeline.fit(X, y)
    with open(output, "wb") as f:
        pkl.dump(model_pipeline, f)


def main():
    wrappers = get_data_pipelines()

    for wrapper in wrappers:
        print(f"fitting {wrapper.name}")
        create_model(
            raw_data_full_path,
            wrapper.pipeline,
            os.path.join(model_path, (wrapper.name + ".pkl")),
            MLPRegressor(**mlp_hyperparams),
            "median_house_value",
        )


if __name__ == "__main__":
    main()
