import numpy as np
import csv

def build_model_from_array(arr):
    """
    From the provided array, build the RandomForestClassifier. Refer to
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    and the Google Colab for more info
    """


if __name__ == "__main__":
    data = np.load("parameters.npy", allow_pickle=True)
    model = build_model_from_array(data)