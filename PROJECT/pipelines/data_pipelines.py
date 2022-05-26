import os
import sys

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.impute import SimpleImputer


class PipelineWrapper:
    def __init__(self, pipeline, name, description=""):
        self.pipeline = pipeline
        self.name = name
        self.description = description


def _ordinal_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "ordinal",
                ColumnTransformer(
                    [
                        (
                            "ordinal",
                            OrdinalEncoder(
                                categories=[
                                    [
                                        "ISLAND",
                                        "<1H OCEAN",
                                        "NEAR BAY",
                                        "NEAR OCEAN",
                                        "INLAND",
                                    ]
                                ]
                            ),
                            ["ocean_proximity"],
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return PipelineWrapper(pipeline, "ordinal")


def _onehot_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            (
                "onehot",
                ColumnTransformer(
                    [("onehot", OneHotEncoder(), ["ocean_proximity"])],
                    remainder="passthrough",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return PipelineWrapper(pipeline, "onehot")


def _onehot_power() -> Pipeline:
    pipeline = Pipeline(
        [("onehot", _onehot_pipeline().pipeline), ("power", PowerTransformer())]
    )
    return PipelineWrapper(pipeline, "onehot_power_transformation")


def _ordinal_power() -> Pipeline:
    pipeline = Pipeline(
        [("ordinal", _ordinal_pipeline().pipeline), ("power", PowerTransformer())]
    )
    return PipelineWrapper(pipeline, "ordinal_power_transformation")


def get_data_pipelines() -> list:
    pipelines = []
    pipelines.append(_onehot_power())
    pipelines.append(_ordinal_power())
    pipelines.append(_ordinal_pipeline())
    pipelines.append(_onehot_pipeline())
    return pipelines
