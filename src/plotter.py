""" This script is intended to take the processed pandas DataFrames, sklearn transformers, opentSNE transformers, and create a bokeh plot inside a jupyter notebook.
"""

from bokeh import palettes
from bokeh.models import ColumnDataSource, HoverTool, Label, LabelSet
from bokeh.plotting import figure, output_file, save, show
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Any


def create_bokeh_plot(
    tsne_transformed_df: pd.DataFrame,
    n_clusters: int = 12,
    kmeans_random_state: int = 30,
    sample_size: int = 200,
    random_state: int = 313,
) -> figure:
    """
    This function takes in the tSNE transformed pandas DataFrame concatenated with the important ingredients, calculates the window size for the graph, uses the kmeans fitted estimator to predict zones inside the graph, creates centroids for each kmeans region, then createa a Bokeh plot with cuisine labels, ingredients, dots for each recipe, square pins for each centroid, image regions for each kmeans region, and hover for the labels.

    Args:
        pd.DataFrame
        sklearn Estimator

    Returns
        Bokeh figure
    """

    random_200 = tsne_transformed_df.sample(sample_size, random_state=random_state)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = random_200["x"].min() - 1, random_200["x"].max() + 1
    y_min, y_max = random_200["y"].min() - 1, random_200["y"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    kmeans = KMeans(n_clusters=n_clusters, random_state=kmeans_random_state).fit(
        random_200.drop(["cuisine_name", "cuisine_id_num"], axis=1)
    )

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    centroids = kmeans.cluster_centers_

    kebab = ColumnDataSource(random_200)
    centroids_cds = ColumnDataSource(pd.DataFrame(data=centroids, columns=["x", "y"]))

    HOVER_TOOLTIPS = [
        ("Cuisine", "@cuisine_name"),
        ("Ingredients", "@important_ingredients"),
    ]

    p = figure(title="KMeans, tSNE, Bokeh", tooltips=HOVER_TOOLTIPS)
    r = p.dot(x="x", y="y", size=15, source=kebab, color="black")

    p.hover.renderers = [r]

    p.square_pin(
        centroids_cds.data["x"],
        centroids_cds.data["y"],
        size=20,
        color="white",
        fill_color=None,
        line_width=4,
    )
    p.image(
        image=[Z],
        x=xx.min(),
        y=xx.min(),
        dw=xx.max() - xx.min(),
        dh=yy.max() - xx.min(),
        palette="Category20_20",
        level="image",
    )

    return p
