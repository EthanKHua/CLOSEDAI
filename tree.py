import numpy as np

class SplitNode:
    # the data and the corresponding label. these should have the same shape
    X = None # this is None if this node has children
    t = None # this is None if this node has children

    # children
    left = None # these are either both None or both exist
    right = None
    feature = -1 # index of feature to split on
    value = 0 # value to split at

    # prediction
    pred = None # only exists if this node is a leaf node

    def __init__(self, X, t):
        """
        X: array-like with shape (N, d)
        t: array-like with shape (N,)

        instantiates SplitNode with the data provided. Always starts off as a leaf node until split() is called.
        """
        self.X = np.array(X)
        self.t = np.array(t)

        if self.X.shape[1] != self.t.shape[0]:
            raise Exception("Shapes", self.X.shape, "and", self.t.shape, "of X and t are not compatible")
        
        pred = np.bincount(t).argmax()
        

    def split(self, index, value):
        """
        Splits the data on the feature at the given index at the given value
        """
        left_indices = self.X[index] <= value
        right_indices = self.X[index] > value
        X_left = self.X[left_indices]
        X_right = self.X[right_indices]
        t_left = self.t[left_indices]
        t_right = self.t[right_indices]

        X = None
        t = None
        pred = None

        left = SplitNode(X_left, t_left)
        right = SplitNode(X_right, t_right)
        self.feature = index
        self.value = value