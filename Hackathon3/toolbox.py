# Pure Python
from collections import Counter
from typing import Dict, List, Tuple, Union

# Plots
import matplotlib.pyplot as plt
import numpy as np

# Numerical
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud


def word_cloud(occurences: Union[List[str], Counter], title: str = None):
    """
    Plot a word cloud based on a list of occurences, or a counter.
    The more an item appears in the list, the bigger it will be displayed.
    """
    if not isinstance(occurences, Counter):
        freqs = Counter(occurences)
    else:
        freqs = occurences

    plt.figure(figsize=(12, 15))
    if title:
        plt.title(title)

    wc = WordCloud(
        max_words=1000, background_color="white", random_state=1, width=1200, height=600
    ).generate_from_frequencies(freqs)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def train_val_indices(
    size: int, val_frac: float = 0.25, seed=None
) -> Dict[str, np.ndarray]:
    """
    Generate train and val indices.

    :param size: the dataset size (number of entries)
    :param frac: the fraction of the dataset that will serve as a validation set
    :param seed: random seed used to generate the indices
    """
    assert 0 <= val_frac <= 1.0, "val_frac must be between 0 and 1"
    indices = np.arange(size)
    n = size - int(size * val_frac)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return indices[:n], indices[n:]


def biplot_visualization(
    pca: PCA,
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    columns: List[str] = None,
):
    """
    Plot a biplot graph: the scaled data after applying a 2D PCA with loadings in vector forms.

    :param pca: PCA object
    :param X: a n by m matrix (or DataFrame), containing the input prior to the PCA transformation
    :param y: a vector of length n containing the target
    :param columns: a list of length m contained the names of the columns
        If not given, X.columns will be used
    """

    columns = columns if columns is not None else X.columns

    X /= X.max(axis=0) - X.min(axis=0)

    df = pd.DataFrame(data=X, columns=["PC1", "PC2"])

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings = pd.DataFrame(loadings, columns=["PC1", "PC2"], index=columns)

    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color=y,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )

    fig.update_layout(
        annotations=[
            dict(
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                x=0,
                y=0,
                text=index,
                showarrow=True,
                ax=row.PC1,
                ay=row.PC2,
                arrowhead=0,
                arrowcolor="#636363",
            )
            for index, row in loadings.iterrows()
        ],
        width=600,
        height=600,
        xaxis_range=[-1, 1],
        yaxis_range=[-1, 1],
    )

    return fig


def accuracy_metric(y_true: np.ndarray, y_pred: List[Dict[str, float]], tol: float = 1e-3):
    """
    Return the accuracy vector between y_true and y_pred.

    :param y_true: an array of str of size n.
    :param y_pred: a list of size n, where each entry contains the probabilities (dict)
        of each variant name to be observed. If not present, a null probability is assumed.
    :param tol: a tolerance for probabilities that must sum to 1.0.
    :return: an array of accuracies of size n.
    """
    acc =[]
    # For each entry
    for true, pred in zip(y_true, y_pred):
        # Safe check that probabilities sum to 1.0 :)
        assert 1.0 - tol <= sum(variant_probability for variant_probability in pred.values()) <= 1.0 + tol
        
        acc.append(pred.get(true, 0.0))

    return np.array(acc)
