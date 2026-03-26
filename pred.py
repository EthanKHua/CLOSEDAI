import numpy as np
import csv
from tree import DecisionTreeClassifier




def build_model_from_array(arr):
    """
    From the provided array, build and return a fitted DecisionTreeClassifier. Refer to
    https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    and the Google Colab for more info
    """
    tree_nodes = [DecisionTreeClassifier() for _ in range(arr[0])]

    stack = [0]
    while stack:
        curr_node_index = stack.pop()
        curr_node = tree_nodes[curr_node_index]
        # if leaf node, update just the value
        if arr[1][curr_node_index] == -1 and arr[2][curr_node_index] == -1:
            curr_node.pred = np.argmax(arr[5][curr_node_index])
            continue

        curr_node.left = tree_nodes[arr[1][curr_node_index]]
        stack.append(arr[1][curr_node_index])

        curr_node.right = tree_nodes[arr[2][curr_node_index]]
        stack.append(arr[2][curr_node_index])

        curr_node.feature = arr[3][curr_node_index]
        curr_node.threshold = arr[4][curr_node_index]




if __name__ == "__main__":
    data = np.load("parameters.npy", allow_pickle=True)
    model = build_model_from_array(data)