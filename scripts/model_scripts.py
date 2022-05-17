from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

import pickle as pkl

import dalex as dx

import pandas as pd


def get_models(names: list = ["gradient_boosting", "neural_network", "random_forest"]):
    models = []

    for name in names:
        with open("../../resources/models/" + name + ".pkl", "rb") as file:
            models.append(pkl.load(file))

    return models


def model_evaluation(models: list, X: pd.DataFrame, y: pd.DataFrame, names: list):
    explainers = []

    for model, name in zip(models, names):
        explainers.append(dx.Explainer(model, X, y, label=name))

    performances = []

    for explainer in explainers:
        performances.append(explainer.model_performance(model_type="regression").result)

    performance = pd.concat(performances)
    performance.sort_values(by="r2", ascending=False, inplace=True)

    return performance
