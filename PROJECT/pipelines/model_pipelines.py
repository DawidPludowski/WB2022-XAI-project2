import os

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

import pandas as pd
import pickle as pkl

from data_pipelines import get_data_pipelines
from config import train_data_full_path, model_path, models_params


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

    os.chdir("PROJECT")

    for wrapper in wrappers:
        for name, model in models_params.items():
            print(f"fitting {name} + {wrapper.name}")
            create_model(
                train_data_full_path,
                wrapper.pipeline,
                os.path.join(model_path, (name + "_" + wrapper.name + ".pkl")),
                model["model"](**model["params"]),
                "median_house_value",
            )


if __name__ == "__main__":
    main()
