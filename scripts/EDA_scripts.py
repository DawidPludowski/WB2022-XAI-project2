import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix


def tmp():
    print("test")


def plot_califormia_map(housing: pd.DataFrame):
    fig = plt.figure(dpi=100, figsize=(4, 4))
    ax = fig.add_axes([1, 1, 1, 1])

    california_img = mpimg.imread(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/California_Locator_Map.PNG/280px-California_Locator_Map.PNG"
    )
    housing.plot(
        kind="scatter",
        x="longitude",
        y="latitude",
        figsize=(10, 7),
        ax=ax,
        s=housing["population"] / 100,
        label="Population",
        c="median_house_value",
        cmap=plt.get_cmap("jet"),
        colorbar=False,
        alpha=0.4,
    )
    plt.imshow(
        california_img,
        extent=[-124.55, -113.80, 32.45, 42.05],
        alpha=0.5,
        cmap=plt.get_cmap("jet"),
    )
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)

    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar()
    cbar.ax.set_yticklabels(
        ["$%dk" % (round(v / 1000)) for v in tick_values], fontsize=14
    )
    cbar.set_label("Median House Value", fontsize=16)

    plt.legend(fontsize=16)
    plt.show()


def plot_correlations(housing: pd.DataFrame):
    corr = housing.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(dpi=100)
    plt.title("Correlation Analysis")
    sns.heatmap(
        corr, mask=mask, annot=False, lw=0, linecolor="white", cmap="PiYG", fmt="0.2f"
    )
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()


def plot_highly_correlated_variables(housing: pd.DataFrame):
    attributes = [
        "median_income",
        "total_rooms",
        "housing_median_age",
    ]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
