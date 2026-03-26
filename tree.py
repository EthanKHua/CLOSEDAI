from collections import Counter

import numpy as np

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
    max_features = None # num of features to consider for each split
    criterion = 'gini' # default gini

    # tree structure
    left = None
    right = None

    # prediction
    pred = None # this has a value if this node is a leaf node

    # splitting parameters
    feature = None # index of feature to split at
    split = None # value to split at

    def __init__(self, max_depth, min_samples_split, max_features=None, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion


    def __gini(self, t):
        """
        Computes the gini impurity of label array t

        Gini = 1 - sum(p_k^2) for each class k

        This measures the the probability that two randomly chosen samples from the node would have different labels 
        Note: Gini = 0 means pure node.
 
        t: array-like of shape (N,)
        Returns: float
        """
        n = len(t)
        if n == 0:
            return 0.0
        _, counts = np.unique(t, return_counts=True)

        probs = counts / n
        return 1.0 - np.sum(probs ** 2)
    

    def __entropy(self, t):
        """
        Computes the entropy/information gain of label array t

        Entropy = -sum(p_k * log2(p_k)) for each class k.

        This measures the average number of bits needed to encode the class label. 
        Note: Entropy = 0 means pure node. 
        
        When we pick splits, we maximize information gain = parent_entropy - weighted_child_entropy,
        which is also equivalent to minimizing the weighted child entropy.

        t: array-like of shape (N,)
        Returns: float in [0, log2(K)] where K = number of classes
        """
        n = len(t)
        if n == 0:
            return 0.0
        _, counts = np.unique(t, return_counts=True)

        probs = counts / n
        probs = probs[probs > 0]  # avoid log2(0)
        return -np.sum(probs * np.log2(probs))
    
    def __criterion(self, t):
        """
        Dispatches to the selected impurity measure

        t: array-like of shape (N,)
        Returns: float
        """
        if self.criterion == 'entropy':
            return self.__entropy(t)
        else:
            return self.__gini(t)

    def __best_split(self, X, t):
        """
        Finds the best (feature, threshold) pair that minimizes weighted gini impurity of the resulting child nodes

        Note that each call only a ranomd subset of max_features features is considered (decorrelation?)
 
        X: array-like of shape (N, d)
        t: array-like of shape (N,)
        Returns: (best_feature_index, best_threshold, best_score) or (None, None, inf)
        """
        N, d = X.shape
        best_feature = None
        best_threshold = None
        best_score = np.inf

        # If no given max_features,randomly pick that many feature indices
        if self.max_features is not None and self.max_features < d:
            feature_indices = np.random.choice(d, size=self.max_features, replace=False)
        else:
            feature_indices = np.arange(d)
        
        for feature_idx in feature_indices:
            col = X[:, feature_idx]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue

            # Use midpoint
            thresholds = (unique_vals[:-1] + unique_vals[1:] / 2.0)

            # subsample threshold to 50 if more than 50 unique midpoints
            if len(thresholds) > 50:
                idx = np.linspace(0, len(thresholds) - 1, 50, dtype=int) # pick evenly spaced indices 10, 20, 30 etc.
                thresholds = thresholds[idx]

            for thresh in thresholds:
                left_mask = col <= thresh # boolean array
                n_left = np.sum(left_mask)
                n_right = N - n_left
                if n_left < 1 or n_right < 1:
                    continue

                # weighted gini impurity of the split
                score = (n_left * self.__criterion(t[left_mask]) +
                         n_right * self.__criterion(t[~left_mask])) / N
                
                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = thresh

        return best_feature, best_threshold, best_score

    def fit(self, X, t, _depth=0):
        """
        Recursively builds the decision tree.

        Stop (make a leaf) conditions:
        - max_depth reached
        - fewer samples than min_samples_split
        - node is already pure (impurity = 0)
        - no valid split found

        X: array-like of shape (N,d)
        t: array-like of shape (N,)
        _depth : int – current depth (for recursion)
        """
        counts = Counter(t)
        self.pred  = max(counts, key=counts.get)

        # stopping conditions
        if (
            (self.max_depth is not None and _depth >= self.max_depth) or
            len(t) < self.min_samples_split or
            self.__criterion_fn(t) == 0.0 # pure node
        ):
            return # remain a leaf
        
        # find best split
        best_feat, best_thresh, best_score = self.__best_split(X, t)
 
        if best_feat is None or best_score >= self.__criterion_fn(t):
            return # no valid split, remain leaf
        
        # record split parameters
        self.feature = best_feat
        self.split = best_thresh

        # partition data
        mask = X[:, best_feat] <= best_thresh
        X_left, t_left   = X[ mask], t[ mask]
        X_right, t_right = X[~mask], t[~mask]

        # grow children
        self.left = DecisionTreeClassifier(
            self.max_depth, self.min_samples_split,
            self.max_features, self.criterion
        )
        self.left.fit(X_left, t_left, _depth + 1)
 
        self.right = DecisionTreeClassifier(
            self.max_depth, self.min_samples_split,
            self.max_features, self.criterion
        )
        self.right.fit(X_right, t_right, _depth + 1)

    def predict(self, x):
        """
        Predicts the label of data point x after fitting
        Traverse tree from root to leaf by comparing x[feature] to the stored
        threshold at each internal node.

        x: array-like with shape (d,)
        """
        # if leaf node, return stored majority prediction
        if self.left is None and self.right is None:
            return self.pred
        
        # if internal node, compare feature to threshold and go left or right
        if x[self.feature] <= self.split:
            return self.left.predict(x)
        else:
            return self.right.predict(x)
        

    def score(self, X_test, t_test):
        """
        Predicts the labels in X_test and returns the accuracy in comparision to t_test
        X_test: array-like with shape (M, d)
        t_test: array-like with shape (M,)
        """
        pass