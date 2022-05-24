import os

data_path = "resources/data"
model_path = "resources/models"

raw_data_full_path = os.path.join(data_path, "housing.csv")

mlp_hyperparams = {
    "hidden_layer_sizes": (10, 100, 20),
    "random_state": 420,
    "max_iter": 1000,
}
