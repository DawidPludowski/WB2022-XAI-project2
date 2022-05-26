import imp
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

data_path: str = "resources/data"
model_path: str = "resources/models"

raw_data_full_path: str = os.path.join(data_path, "housing.csv")
train_data_full_path: str = os.path.join(data_path, "housing_train.csv")
test_data_full_path: str = os.path.join(data_path, "housing_test.csv")

models_params: dict = {
    "random_forest": {
        "model": RandomForestRegressor,
        "params": {
            "criterion": "absolute_error",
            "max_depth": 9,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "n_estimators": 300,
        },
    },
    "neural_network": {
        "model": MLPRegressor,
        "params": {"hidden_layer_sizes": (10, 100, 20), "random_state": 2138},
    },
    "gradient_boosting": {
        "model": GradientBoostingRegressor,
        "params": {"criterion": "mse", "max_features": "auto"},
    },
    "linear_regression": {"model": LinearRegression, "params": {}},
    "decision_tree": {
        "model": DecisionTreeRegressor,
        "params": {
            "criterion": "absolute_error",
            "max_features": "auto",
            "min_samples_split": 8,
        },
    },
}
