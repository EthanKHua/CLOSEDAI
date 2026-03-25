from tree import TreeNode

# first implement decision tree to implement random forest
class DecisionTreeClassifier:
    """
    The decision tree supports the following operations:
    - fit(): fits the model to the data provided
    - predict(): after fitting, predict the label of a data point
    - score(): after fitting, returns the model's accuracy on a given labelled dataset

    The DecisionTreeClassifier can be thought of as a single node which is either an internal node
    or a leaf node.
    """
    # hyperparameters
    max_depth = None
    min_samples_split = None

    # tree structure
    left = None
    right = None

    # prediction
    pred = None # this has a value if this node is a leaf node

    # splitting parameters
    feature = None # index of feature to split at
    split = None # value to split at

    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split


    def __criterion(self, X, t, feature, split):
        """
        Helper function that outputs the cross entropy (or gini impurity?) of the data split provided
        X: array-like of shape (N,d)
        t: array-like of shape (N,)
        feature: index of feature to split at
        split: value to split at (i.e. we split X by the condition X[feature] <= split)
        """
        pass


    def fit(self, X, t):
        """
        X: array-like of shape (N,d)
        t: array-like of shape (N,)
        """
        pass


    def predict(self, x):
        """
        Predicts the label of data point x after fitting
        x: array-like with shape (d,)
        """
        pass


    def score(self, X_test, t_test):
        """
        Predicts the labels in X_test and returns the accuracy in comparision to t_test
        X_test: array-like with shape (M, d)
        t_test: array-like with shape (M,)
        """
        pass